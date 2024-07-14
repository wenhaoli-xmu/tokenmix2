import torch
from src.modifier import Modifier
import random

from src.modifiers.modify_llama import segment
from src.modifiers.modify_llama_hie5dec import Decoder, CausalLMOutputWithPast


def kmp_search(lst, sublst):
    def compute_lps(sublst):
        m = len(sublst)
        lps = [0] * m
        j = 0
        for i in range(1, m):
            while j > 0 and sublst[i] != sublst[j]:
                j = lps[j - 1]
            if sublst[i] == sublst[j]:
                j += 1
            lps[i] = j
        return lps

    lps = compute_lps(sublst)
    m, n = len(sublst), len(lst)
    i, j = 0, 0
    while i < n:
        if sublst[j] == lst[i]:
            i += 1
            j += 1
        if j == m:
            return i - m
        elif i < n and sublst[j] != lst[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return -1


class Model(torch.nn.Module):
    def __init__(
            self, 
            decoder: Decoder, 
            chunk_size: int):

        super().__init__()
        self.decoder = decoder
        self.chunk_size = chunk_size

    def ft_params(self):
        return self.decoder.ft_params()

    def reset(self):
        self.decoder.reset()
        
    def forward(
            self, 
            input_ids, 
            labels=None,
            local_rank=-1,
            **kwargs
        ):
        label_exists = labels is not None

        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0,1)
        if labels is not None and labels.ndim == 3:
            labels = labels.flatten(0,1)

        if local_rank >= 0:
            device = local_rank
        else:
            device = next(self.decoder.parameters()).device

        input_ids = input_ids.cuda(device)
        input_ids_chunk = list(segment(input_ids, dim=-1, n=self.chunk_size))
        input_ids_prefill = input_ids_chunk[:-1]

        if len(input_ids_prefill) > 0:
            memory = self.decoder(input_ids=torch.cat(input_ids_prefill, dim=0), prefill=True)
            memory = memory.chunk(memory.shape[1], dim=1)
        else:
            memory = []
        
        logits = []

        if label_exists:

            # =============================================================
            # NOTE: test
            passkey_length = (labels != -100).sum().item()
            passkey = input_ids[:,-passkey_length:].ravel().tolist()
            input_ids_list = input_ids[:,:-passkey_length].ravel().tolist()
            position = kmp_search(input_ids_list, passkey)
            # =============================================================

            labels = list(segment(labels, dim=-1, n=self.chunk_size))
            losses = []
            length = []
        else:
            labels = [None] * len(input_ids_chunk)

        for chunk_id, (chunk_inputs, chunk_labels) in enumerate(zip(input_ids_chunk, labels)):
            history = torch.cat(memory[:chunk_id], dim=1) if chunk_id > 0 and len(memory) > 0 else None
            outputs = self.decoder.forward(chunk_inputs, labels=chunk_labels, memory=history)

            logits.append(outputs.logits)

            if label_exists and not outputs.loss.isnan():
                length.append((chunk_labels != -100).sum().item())
                losses.append(outputs.loss * length[-1])

        logits = torch.cat(logits, dim=-2)

        if label_exists:
            losses = sum(losses)
            loss = losses / sum(length)
            # ==========================
            # NOTE: test
            print(loss.item(), position)
            # ==========================

        else:
            loss = None
        
        return CausalLMOutputWithPast(logits=logits, loss=loss)


class LlamaHIE6(Modifier):
    def __init__(self, model, save_ckp, load_ckp, config):
        self.get_conf(config)
        assert isinstance(self.conf, dict)
        chunk_size = self.conf["chunk_size"]
        enable_lora = False
        lora_kwargs = self.conf["lora_kwargs"]
        
        decoder = Decoder(
            model, 
            chunk_size=chunk_size,
            enable_lora=enable_lora,
            lora_kwargs=lora_kwargs)

        decoder = Model(
            decoder, 
            chunk_size=chunk_size)

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
        kv_caches = self.model.decoder(input_ids=context_ids, prefill=True)
        assert kv_caches.ndim == 6 and kv_caches.shape[0] == 2
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
