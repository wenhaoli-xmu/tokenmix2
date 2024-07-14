import torch
from ..modifier import Modifier
from .modify_llama import segment
from .modify_llama_hybird9dec import Decoder
import random


class Model(torch.nn.Module):
    def __init__(
            self, 
            decoder: Decoder, 
            chunk_size: int,
            history: int,
            reduction: dict,
        ):
        super().__init__()
        self.decoder = decoder
        self.chunk_size = chunk_size
        self.history = history
        self.reduction = reduction


    def ft_params(self):
        params = self.decoder.ft_params()
        return params


    def reset(self):
        self.decoder.reset()


    def forward(
            self,
            input_ids,
            labels=None,
            local_rank=None,
            **kwargs
        ):

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.int64)[None, :]
            labels = torch.tensor(labels, dtype=torch.int64)[None, :]

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

            if self.history > 0:
                prefix_chunk = torch.zeros((1, self.history), dtype=prefil_chunk.dtype, device=prefil_chunk.device)
                shifts_chunk = torch.cat([prefix_chunk, prefil_chunk[:-1, -self.history:]], dim=0)
                prefil_chunk = torch.cat([shifts_chunk, prefil_chunk], dim=-1)

            kv_caches = self.decoder(input_ids=prefil_chunk, prefill=True)
            kv_caches = [[layer_cache[..., self.history:, :].transpose(0,1).flatten(1,2).unsqueeze(0) 
                for layer_cache in cache]
                for cache in kv_caches]
        else:
            kv_caches = None
    
        is_reduced = self.reduction["enable_random"] and labels is not None
        if is_reduced:
            assert labels.shape[0] == 1

            # random select a chunk for training
            chunk_labels = list(segment(labels, dim=-1, n=self.chunk_size))
            valid_tokens = [(x != -100).sum().item() for x in chunk_labels]
            sum_valid_tokens = sum(valid_tokens)

            if sum_valid_tokens > 0:
                select_prob = [x / sum_valid_tokens for x in valid_tokens]
                choice = random.choices(
                    list(range(len(select_prob))), 
                    weights=select_prob, 
                    k=1)[0]
            else:
                choice = random.choices(
                    list(range(len(valid_tokens))),
                    k=1)[0]

            # truncate the kv caches
            new_key_caches = []
            new_val_caches = []

            if kv_caches is not None:
                for key_cache, val_cache in zip(kv_caches[0], kv_caches[1]):
                    key_cache = key_cache[..., :choice * self.chunk_size, :]
                    val_cache = val_cache[..., :choice * self.chunk_size, :]
                    new_key_caches.append(key_cache)
                    new_val_caches.append(val_cache)
                kv_caches = [new_key_caches, new_val_caches]

            # truncate the input indices & labels
            input_ids = input_ids[:, choice * self.chunk_size: (choice + 1) * self.chunk_size]
            labels = labels[:, choice * self.chunk_size: (choice + 1) * self.chunk_size]

        outputs = self.decoder(input_ids, kv_caches=kv_caches, labels=labels, is_reduced=is_reduced)
        return outputs


class LlamaHybird9(Modifier):
    def __init__(self, model, save_ckp, load_ckp, config):
        self.get_conf(config)
        assert isinstance(self.conf, dict)
        chunk_size = self.conf["chunk_size"]
        enable_lora = self.conf["enable_lora"]
        lora_kwargs = self.conf["lora_kwargs"]
        history = self.conf['history']
        retrieval = self.conf['retrieval'] if 'retrieval' in self.conf else None
        reduction = self.conf['reduction'] if 'reduction' in self.conf else {"enable_random": False, "max_chunk": None}
        attn_mode = self.conf['attn_mode'] if 'attn_mode' in self.conf else "legacy"
        
        assert retrieval in ('avgpool', 'eos_token', None)
        assert attn_mode in ('legacy', 'fast')

        self.chunk_size = chunk_size
        
        decoder = Decoder(
            model, 
            chunk_size=chunk_size,
            enable_lora=enable_lora,
            lora_kwargs=lora_kwargs,
            retrieval=retrieval,
            attn_mode=attn_mode)

        decoder = Model(
            decoder, 
            chunk_size=chunk_size,
            history=history,
            reduction=reduction)

        super().__init__(decoder, save_ckp, load_ckp)


    def ft_params(self):
        return self.model.ft_params()


    def reset(self):
        self.model.reset()


    def prefill(self, input_ids):
        device = next(iter(self.model.decoder.parameters())).device
        input_ids = input_ids.to(device)

        input_chunks = list(segment(input_ids, dim=-1, n=self.chunk_size))
        prefil_chunk = torch.cat(input_chunks[:-1], dim=0)
        kv_caches = self.model.decoder(input_ids=prefil_chunk, prefill=True)

        return kv_caches


    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=128, eos_token_id=[2]):

        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0,1)

        # put the tensor on to the model's device
        device = next(iter(self.model.parameters())).device
        input_ids = input_ids.to(device)

        # compute how many chunks are there to prefill
        input_length = input_ids.shape[-1]
        num_chunks = input_length // self.chunk_size
        if input_length % self.chunk_size == 0:
            num_chunks -= 1
        
        # seperate
        context_ids = input_ids[:,:num_chunks * self.chunk_size]
        remain_ids = input_ids[:,num_chunks * self.chunk_size:-1]
        newest_ids = input_ids[:,-1:]

        """prefilling stage"""
        if num_chunks > 0:
            prefill_chunks = torch.cat(list(segment(context_ids, dim=-1, n=self.chunk_size)), dim=0)
            kv_caches = self.model.decoder(input_ids=prefill_chunks, prefill=True)
            kv_caches = [[layer_cache.transpose(0,1).flatten(1,2).unsqueeze(0) 
                for layer_cache in cache]
                for cache in kv_caches]
        else:
            kv_caches = None

        if remain_ids.shape[-1] > 0:
            self.model.decoder(input_ids=remain_ids, kv_caches=kv_caches, generation=True)

        """generation stage"""
        while input_ids.shape[-1] <= input_length + max_new_tokens:
            logits = self.model.decoder(input_ids=newest_ids, kv_caches=kv_caches, generation=True).logits
            newest_ids = logits.argmax(dim=-1)
            if newest_ids.ravel().item() in eos_token_id:
                break
            newest_ids = newest_ids.to(input_ids.device)
            input_ids = torch.cat([input_ids, newest_ids], dim=-1)

        self.model.reset()

        return input_ids
