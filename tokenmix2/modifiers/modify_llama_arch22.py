import torch
import torch.utils.checkpoint
from src.modifier import SegmentRecurrentModifier
from copy import deepcopy
from typing import List
from functools import partial
import random

from src.modifiers.modify_llama import segment
from src.modifiers.modify_llama_arch22enc import Encoder
from src.modifiers.modify_llama_arch22dec import Decoder


class Model(torch.nn.Module):
    def __init__(
            self, 
            encoder: Encoder, 
            decoder: Decoder, 
            chunk_size: int, 
            eos_token: torch.Tensor
        ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.chunk_size = chunk_size

    def ft_params(self):
        params = self.encoder.ft_params() + self.decoder.ft_params()
        return params

    def reset(self):
        self.decoder.reset()

    def update_memory(
            self, 
            input_ids, 
            **kwargs
        ):
        memory = self.encoder(
            input_ids=input_ids, 
        ) if input_ids.shape[-1] == self.chunk_size else None
        self.decoder.reset()
        return memory
        
    def forward(
            self, 
            input_ids, 
            memory=None, 
            labels=None,
            local_rank=-1,
            num_trainble_chunk=8,
            **kwargs
        ):
        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0,1)
        if labels.ndim == 3:
            labels = labels.flatten(0,1)

        input_ids = input_ids.cuda(local_rank)

        # chunk the input_ids tensor
        input_chunks = list(segment(input_ids, dim=-1, n=self.chunk_size))
        prefil_chunk = input_chunks[:-1]
        remain_chunk = input_chunks[-1]
        num_prefil_chunk = len(prefil_chunk)
        prefil_chunk = torch.cat(prefil_chunk, dim=0)
        labels = labels[...,-remain_chunk.shape[-1]:]

        # build the select mask
        train_idx = random.sample(range(0, num_prefil_chunk), k=num_trainble_chunk)
        select_mask = [1 if x in train_idx else 0 for x in range(num_prefil_chunk)]
        select_mask = torch.tensor(select_mask, dtype=torch.bool, device=input_ids.device)
        
        memories = [None for _ in range(num_prefil_chunk)]

        with torch.enable_grad():
            chunks = prefil_chunk[select_mask, ...]
            memory = self.update_memory(input_ids=chunks)
            memory = torch.chunk(memory, chunks=num_trainble_chunk, dim=0)
            for idx, mem in zip(train_idx, memory):
                assert mem.requires_grad
                memories[idx] = mem

        with torch.no_grad():
            chunks = prefil_chunk[~select_mask, ...]
            memory = self.update_memory(input_ids=chunks)
            memory = torch.chunk(memory, chunks=num_prefil_chunk - num_trainble_chunk, dim=0)
            for mem in memory:
                assert not mem.requires_grad
                idx = memories.index(None)
                memories[idx] = mem

        memory = torch.cat(memories, dim=-2)
        outputs = self.decoder.forward(remain_chunk, memory=memory, labels=labels)
        return outputs


class LlamaARCH22(SegmentRecurrentModifier):
    def __init__(self, model, save_ckp, load_ckp, config):
        self.get_conf(config)
        assert isinstance(self.conf, dict)
        chunk_size = self.conf["chunk_size"]
        enable_lora = self.conf["enable_lora"]
        lora_kwargs = self.conf["lora_kwargs"]

        encoder = deepcopy(model)
        decoder = model
        eos_token = encoder.model.embed_tokens.weight[2,:]

        encoder = Encoder(
            chunk_size=chunk_size, 
            enable_lora=enable_lora, 
            lora_kwargs=lora_kwargs)
        
        decoder = Decoder(
            decoder, 
            chunk_size=chunk_size)

        encoder_decoder = Model(
            encoder, 
            decoder, 
            chunk_size=chunk_size, 
            eos_token=eos_token)

        super().__init__(encoder_decoder, save_ckp, load_ckp, chunk_size=chunk_size)

    def ft_params(self):
        return self.model.ft_params()

    def reset(self):
        self.model.reset()

    def get_memories(self, segment_id):
        ...

    @torch.no_grad()
    def generate(
            self, 
            input_ids: torch.Tensor,
            max_new_tokens: int = 256,
            eos_token_id: List = [2],
            tokenizer = None,
    ) -> List:
        
        prompt_length = input_ids.shape[1]
        context = input_ids[:,:-1]
        chunker = partial(segment, dim=-1, n=self.chunk_size)

        """
        memory & kv cache injection
        """
        past_memory = None

        for chunk_context in chunker(context):
            if chunk_context.shape[-1] == self.chunk_size:
                past_memory = self.model.update_memory(input_ids=chunk_context, memory=past_memory)
            else:
                self.model(input_ids=chunk_context)
    
        """
        token by token generation
        """
        new_token = input_ids[:,-1:]
        while input_ids.shape[1] < prompt_length + max_new_tokens:
            logits = self.model(input_ids=new_token, memory=past_memory).logits.cpu()

            # ============================================
            # NOTE: for test
            scores, words = torch.topk(logits, k=20, dim=-1)
            scores, words = scores.tolist()[0][0], words.tolist()[0][0]

            generated = tokenizer.decode(input_ids[:,prompt_length:][0])
            generated = generated.replace('\n', '\\n')
            print(f"index: {input_ids.shape[-1] - prompt_length}  {generated}")
            for idx, (score, index) in enumerate(zip(scores, words)):
                word = tokenizer.decode(index)
                print(f"\t{idx}. {word} {score}".replace('\n', '\\n'))
            choice = input(">> ")
            if choice == '':
                choice = 0
            choice = int(choice)
            new_token = torch.tensor([[words[choice]]])
            # ============================================
            
            # new_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, new_token.to(input_ids.device)], dim=1)
            if new_token.item() in eos_token_id:
                break

            if input_ids.shape[-1] % self.chunk_size == 0:
                past_memory = self.model.update_memory(input_ids=input_ids[:,-self.chunk_size:], memory=past_memory)

        self.reset()
        return input_ids
