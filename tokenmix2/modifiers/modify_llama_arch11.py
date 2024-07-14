import torch
import torch.utils.checkpoint
from src.modifier import SegmentRecurrentModifier
from copy import deepcopy
from typing import List
from functools import partial

from src.modifiers.modify_llama import segment
from src.modifiers.modify_llama_arch11enc import Encoder
from src.modifiers.modify_llama_arch11dec import Decoder


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

        self.init_cells = torch.zeros_like(eos_token)[None,None,:].expand(32,chunk_size,-1)
        self.init_state = torch.zeros_like(eos_token)[None,None,:].expand(32,chunk_size,-1)

        self.init_memory = torch.zeros_like(eos_token)[None,None,:].expand(64,chunk_size,-1)
        self.i_gate = torch.nn.Parameter(eos_token[None,None,:], requires_grad=True)
        self.f_gate = torch.nn.Parameter(eos_token[None,None,:], requires_grad=True)
        self.o_gate = torch.nn.Parameter(eos_token[None,None,:], requires_grad=True)
        self.g_gate = torch.nn.Parameter(eos_token[None,None,:], requires_grad=True)

    def ft_params(self):
        params = self.encoder.ft_params() + self.decoder.ft_params()
        params += [self.i_gate, self.f_gate, self.o_gate, self.g_gate]
        return params

    def reset(self):
        self.decoder.reset()

    def update_memory(
            self, 
            input_ids,
            memory=None,
            cells=None,
            state=None,
            use_mem=True, 
            **kwargs
        ):

        """
        encoder forward
        """
        if use_mem and memory is not None:
            assert memory.ndim == 3 and memory.shape[0] == 64
            cells, state = memory.chunk(2, dim=0)

        cells, state = self.encoder(
            input_ids=input_ids, 
            cells=cells if cells is not None else self.init_cells,
            state=state if state is not None else self.init_state,
            i_gate=self.i_gate.expand(-1,self.chunk_size,-1),
            f_gate=self.f_gate.expand(-1,self.chunk_size,-1),
            o_gate=self.o_gate.expand(-1,self.chunk_size,-1),
            g_gate=self.g_gate.expand(-1,self.chunk_size,-1)
        ) if input_ids.shape[-1] == self.chunk_size else None

        if use_mem:
            return torch.cat([cells, state], dim=0)
        else:
            return cells, state

        
    def forward(
            self, 
            input_ids, 
            state=None,
            memory=None,
            labels=None,
            use_mem=True
        ):
        """
        decoder forward
        """
        if memory is not None:
            memory = memory.chunk(2, dim=0)[1]

        outputs = self.decoder.forward(
            input_ids, 
            memory=memory if use_mem else state, 
            labels=labels)

        return outputs


class LlamaARCH11(SegmentRecurrentModifier):
    def __init__(self, model, save_ckp, load_ckp, config):

        self.get_conf(config)
        chunk_size = self.conf["chunk_size"]
        enable_lora = self.conf["enable_lora"]
        lora_kwargs = self.conf["lora_kwargs"]
        tune_mlp = self.conf["tune_mlp"]

        encoder = deepcopy(model)
        decoder = model
        eos_token = encoder.model.embed_tokens.weight[2,:]

        encoder = Encoder(encoder, chunk_size=chunk_size, tune_mlp=tune_mlp, enable_lora=enable_lora, lora_kwargs=lora_kwargs)
        decoder = Decoder(decoder, chunk_size=chunk_size, enable_lora=enable_lora, lora_kwargs=lora_kwargs)
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
