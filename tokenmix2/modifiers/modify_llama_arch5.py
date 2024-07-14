import torch
import torch.utils.checkpoint
from src.modifier import SegmentRecurrentModifier
from copy import deepcopy
from typing import List
from functools import partial

from src.modifiers.modify_llama import segment
from src.modifiers.modify_llama_arch5enc import Encoder
from src.modifiers.modify_llama_arch5dec import Decoder


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, chunk_size, eos_token):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.chunk_size = chunk_size
        self.init_memory = torch.nn.Parameter(eos_token[None,None,:].expand(-1,self.chunk_size,-1).detach(), requires_grad=True)

    def ft_params(self):
        return self.encoder.ft_params() + self.decoder.ft_params() + [self.init_memory]

    def reset(self):
        self.decoder.reset()

    def update_memory(self, input_ids, memory=None, **kwargs):

        # NOTE：当input_ids长度达标时，使用encoder对memory进行更新得到updated_memory
        updated_memory = self.encoder(
            input_ids=input_ids, 
            memory=memory if memory is not None else self.init_memory
        ) if input_ids.shape[-1] == self.chunk_size else None

        return updated_memory
        
    def forward(self, input_ids, memory=None, labels=None):

        assert input_ids.shape[1] <= self.chunk_size
        assert memory is None or memory.shape == (1, self.chunk_size, 4096)

        outputs = self.decoder.forward(input_ids, memory=memory, labels=labels)

        return outputs


class LlamaARCH5(SegmentRecurrentModifier):
    def __init__(self, model, save_ckp, load_ckp, config):

        self.get_conf(config)
        chunk_size = self.conf["chunk_size"]
        enable_lora = self.conf["enable_lora"]
        lora_kwargs = self.conf["lora_kwargs"]

        encoder = deepcopy(model)
        decoder = model
        eos_token = encoder.model.embed_tokens.weight[2,:]

        encoder = Encoder(encoder, chunk_size=chunk_size, enable_lora=enable_lora, lora_kwargs=lora_kwargs)
        decoder = Decoder(decoder, chunk_size=chunk_size)
        encoder_decoder = Model(encoder, decoder, chunk_size=chunk_size, eos_token=eos_token)

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
            eos_token_id: List = [2]
    ) -> List:
        
        prompt_length = input_ids.shape[1]
        context = input_ids[:,:-1]
        chunker = partial(segment, dim=-1, n=self.chunk_size)

        """
        memory & kv cache injection
        """
        past_memory = None
        past_memory_clone = None

        for chunk_context in chunker(context):
            self.model(input_ids=chunk_context, memory=past_memory)
            past_memory = None

            if chunk_context.shape[-1] == self.chunk_size:
                past_memory = self.model.update_memory(input_ids=chunk_context, memory=past_memory_clone)
                past_memory_clone = past_memory.data.clone()
    
        """
        token by token generation
        """
        new_token = input_ids[:,-1:]
        while input_ids.shape[1] < prompt_length + max_new_tokens:
            logits = self.model(input_ids=new_token, memory=past_memory).logits.cpu()
            past_memory = None
            
            new_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, new_token.to(input_ids.device)], dim=1)
            if new_token.item() in eos_token_id:
                break

            if input_ids.shape[-1] % self.chunk_size == 0:
                past_memory = self.model.update_memory(input_ids=input_ids[:,-self.chunk_size:], memory=past_memory_clone)
                past_memory_clone = past_memory.data.clone()

        self.reset()
        return input_ids
