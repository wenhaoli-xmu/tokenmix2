import torch
from src.modifier import Modifier
from src.modifiers.modify_llama import segment
from src.modifiers.modify_llama_hybird5dec import Decoder

from copy import deepcopy



class Model(torch.nn.Module):
    def __init__(
            self, 
            decoder: Decoder, 
            chunk_size: int,
            trainable_token: int,
            fix_layers: int
        ):
        super().__init__()
        self.decoder = decoder
        self.chunk_size = chunk_size
        self.num_trainable_chunk = trainable_token // chunk_size
        self.fix_layers = fix_layers

        eos_token = deepcopy(self.decoder.model.model.embed_tokens.weight[2,:].data)
        eos_token = eos_token[None, None, :]
        self.sum_token = torch.nn.Parameter(eos_token, requires_grad=True)
        assert self.sum_token.ndim == 3 and self.sum_token.shape[:2] == (1, 1)


    def ft_params(self):
        params = self.decoder.ft_params()
        params.append(self.sum_token)
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
            assert self.sum_token.ndim == 3 and self.sum_token.shape[:2] == (1, 1)

            prefil_chunk = torch.cat(input_chunks[:-1], dim=0)
            kv_caches = self.decoder(input_ids=prefil_chunk, prefill=True, sum_token=self.sum_token)           
            kv_caches = kv_caches.chunk(kv_caches.shape[2], dim=2)
            kv_caches = torch.cat(kv_caches, dim=-2)

            if self.fix_layers > 0:
                true_kv_caches = self.decoder(input_ids=torch.cat(input_chunks[:-1], dim=-1), prefill=True)
                kv_caches[:,:self.fix_layers,...] = true_kv_caches[:,:self.fix_layers,...]
        else:
            kv_caches = None

        # generation phase
        outputs = self.decoder(input_ids, kv_caches=kv_caches, labels=labels)
        return outputs



class LlamaHybird5(Modifier):
    def __init__(self, model, save_ckp, load_ckp, config):
        self.get_conf(config)
        assert isinstance(self.conf, dict)
        chunk_size = self.conf["chunk_size"]
        enable_lora = self.conf["enable_lora"]
        lora_kwargs = self.conf["lora_kwargs"]
        fix_layers = self.conf["fix_layers"]
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
            trainable_token=trainable_token,
            fix_layers=fix_layers)

        super().__init__(decoder, save_ckp, load_ckp)


    def ft_params(self):
        return self.model.ft_params()


    def reset(self):
        self.model.reset()


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
            kv_caches = kv_caches.chunk(kv_caches.shape[2], dim=2)
            kv_caches = torch.cat(kv_caches, dim=-2)
        else:
            kv_caches = None

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
