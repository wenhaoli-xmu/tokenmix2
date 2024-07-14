import torch
from src.modifier import Modifier
import random

from src.modifiers.modify_llama import segment
from src.modifiers.modify_llama_hybird2dec import Decoder, CausalLMOutputWithPast


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
        prefill = input_ids.shape[-1] > self.chunk_size

        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0,1)
        if label_exist and labels.ndim == 3:
            labels = labels.flatten(0,1)

        if rank_exist:
            device = torch.device(local_rank)
        else:
            device = next(iter(self.decoder.parameters())).device
        input_ids = input_ids.to(device)

        # segment the input & labels
        input_chunks = list(segment(input_ids, dim=-1, n=self.chunk_size))

        # prepare prefilling chunks
        if prefill:
            prefil_chunk = torch.cat(input_chunks[:-1], dim=0)
            shift_chunk = torch.cat([torch.zeros((1, self.chunk_size), dtype=torch.int64, device=device), prefil_chunk[:-1]], dim=0)
            concat_chunk = torch.cat([shift_chunk, prefil_chunk], dim=-1)

            # prefil to get kv cache
            kv_caches = self.decoder(input_ids=concat_chunk, prefill=True)
            assert kv_caches.ndim == 6 and kv_caches.shape[0] == 2

            kv_caches = kv_caches.chunk(kv_caches.shape[2], dim=2)
            kv_caches = [cache[...,-self.chunk_size:,:] for cache in kv_caches]
            kv_caches = torch.cat(kv_caches, dim=-2)
        else:
            kv_caches = None

        # compute loss & early return
        if label_exist:
            label_chunks = list(segment(input_ids, dim=-1, n=self.chunk_size))
            total_loss = torch.tensor(0, dtype=torch.bfloat16, device=device)
            total_length = 0

            for chunk_id, (input_chunk, label_chunk) in enumerate(zip(input_chunks, label_chunks)):
                num_kv_cache = self.chunk_size * chunk_id
                chunk_length = input_chunk.shape[-1] - 1
                total_length += chunk_length
                loss = self.decoder(input_chunk, kv_caches=kv_caches[...,:num_kv_cache,:], labels=label_chunk).loss * chunk_length
                total_loss += loss

            total_loss /= total_length

            return CausalLMOutputWithPast(loss=total_loss, logits=None)

        # generation phase
        last_chunk = input_chunks[-1]
        outputs = self.decoder(last_chunk, kv_caches=kv_caches)
        return outputs


class LlamaHybird2(Modifier):
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
