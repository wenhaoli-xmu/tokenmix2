import torch
from src.modifier import Modifier
import random

from src.modifiers.modify_llama import segment
from src.modifiers.modify_llama_hybird1dec import Decoder


class Model(torch.nn.Module):
    def __init__(
            self, 
            decoder: Decoder, 
            chunk_size: int,
            trainable_token: int
        ):
        super().__init__()
        self.decoder = decoder
        self.chunk_size = chunk_size
        self.num_trainable_chunk = trainable_token // chunk_size

    def ft_params(self):
        return self.decoder.ft_params()

    def reset(self):
        self.decoder.reset()
        
    def forward(
            self, 
            input_ids, 
            labels=None,
            local_rank=None,
            **kwargs
        ):
        label_exist = labels is not None
        rank_exist = local_rank is not None

        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0,1)
        if label_exist and labels.ndim == 3:
            labels = labels.flatten(0,1)

        if rank_exist:
            device = torch.device(local_rank)
        else:
            device = next(iter(self.decoder.parameters())).device
        input_ids = input_ids.to(device)

        # chunk the input_ids tensor
        input_chunks = list(segment(input_ids, dim=-1, n=self.chunk_size))
        prefil_chunk = input_chunks[:-1]
        remain_chunk = input_chunks[-1]
        num_prefil_chunk = len(prefil_chunk)
        num_trainable_chunk = min(num_prefil_chunk - 1, self.num_trainable_chunk)
        prefil_chunk = torch.cat(prefil_chunk, dim=0)
        shift_chunk = torch.cat([torch.zeros((1, self.chunk_size), dtype=torch.int64, device=device), prefil_chunk[:-1]], dim=0)
        concat_chunk = torch.cat([shift_chunk, prefil_chunk], dim=-1)

        if label_exist:
            labels = labels[...,-remain_chunk.shape[-1]:]

        # build the select mask
        train_idx = random.sample(range(0, num_prefil_chunk), k=num_trainable_chunk)
        select_mask = [1 if x in train_idx else 0 for x in range(num_prefil_chunk)]
        select_mask = torch.tensor(select_mask, dtype=torch.bool, device=device)
        
        kv_caches = [None for _ in range(num_prefil_chunk)]

        # enable gradient computation
        chunks = concat_chunk[select_mask, ...]
        kv_cache = self.decoder(input_ids=chunks, prefill=True)
        assert kv_cache.ndim == 6 and kv_cache.shape[0] == 2

        kv_cache = torch.chunk(kv_cache, chunks=num_trainable_chunk, dim=2)
        for idx, kv in zip(train_idx, kv_cache):
            kv_caches[idx] = kv

        # disable gradient computation
        with torch.no_grad():
            chunks = concat_chunk[~select_mask, ...]
            kv_cache = self.decoder(input_ids=chunks, prefill=True)
            assert kv_cache.ndim == 6 and kv_cache.shape[0] == 2

            kv_cache = torch.chunk(kv_cache, chunks=num_prefil_chunk - num_trainable_chunk, dim=2)
            for kv in kv_cache:
                idx = kv_caches.index(None)
                kv_caches[idx] = kv

        while None in kv_caches:
            kv_caches.remove(None)
        kv_caches = [cache[...,-self.chunk_size:,:] for cache in kv_cache]
        kv_caches = torch.cat(kv_caches, dim=-2)

        # use generated kv cache for inference
        outputs = self.decoder.forward(remain_chunk, kv_caches=kv_caches, labels=labels)
        return outputs


class LlamaHybird1(Modifier):
    def __init__(self, model, save_ckp, load_ckp, config):
        self.get_conf(config)
        assert isinstance(self.conf, dict)
        chunk_size = self.conf["chunk_size"]
        enable_lora = self.conf["enable_lora"]
        lora_kwargs = self.conf["lora_kwargs"]
        trainable_token = self.conf['trainable_token'] if 'trainable_token' in self.conf else 1024
        use_sdpa = self.conf['use_sdpa'] if 'use_sdpa' in self.conf else False
        self.chunk_size = chunk_size
        
        decoder = Decoder(
            model, 
            chunk_size=chunk_size,
            enable_lora=enable_lora,
            lora_kwargs=lora_kwargs,
            use_sdpa=use_sdpa)

        decoder = Model(
            decoder, 
            chunk_size=chunk_size,
            trainable_token=trainable_token)

        super().__init__(decoder, save_ckp, load_ckp)

    def ft_params(self):
        return self.model.ft_params()

    def reset(self):
        self.model.reset()
