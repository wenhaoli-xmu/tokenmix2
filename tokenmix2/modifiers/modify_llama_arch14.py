import torch
import torch.utils.checkpoint
from ..modifier import SegmentRecurrentModifier
from copy import deepcopy
from typing import List
from functools import partial

from ..modifiers.modify_llama import segment
from ..modifiers.modify_llama_arch14enc import Encoder
from ..modifiers.modify_llama_arch14dec import Decoder


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
        self.i_gate = torch.nn.Parameter(torch.zeros_like(eos_token)[None,None,:].expand(-1,chunk_size,-1).clone(), requires_grad=True)
        self.f_gate = torch.nn.Parameter(torch.zeros_like(eos_token)[None,None,:].expand(-1,chunk_size,-1).clone(), requires_grad=True)
        self.o_gate = torch.nn.Parameter(torch.zeros_like(eos_token)[None,None,:].expand(-1,chunk_size,-1).clone(), requires_grad=True)
        self.g_gate = torch.nn.Parameter(torch.zeros_like(eos_token)[None,None,:].expand(-1,chunk_size,-1).clone(), requires_grad=True)
        
        torch.nn.init.xavier_uniform_(self.i_gate.data)
        torch.nn.init.xavier_uniform_(self.f_gate.data)
        torch.nn.init.xavier_uniform_(self.o_gate.data)
        torch.nn.init.xavier_uniform_(self.g_gate.data)

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
            i_gate=self.i_gate,
            f_gate=self.f_gate,
            o_gate=self.o_gate,
            g_gate=self.g_gate
        ) if input_ids.shape[-1] == self.chunk_size else (None, None)

        self.decoder.reset()
        
        if use_mem:
            return torch.cat([cells, state], dim=0) if cells is not None else None
        else:
            return cells, state

    def forward(
            self, 
            input_ids, 
            cells=None,
            state=None,
            memory=None,
            labels=None,
            use_mem=True,
            update_memory=False,
        ):
        """
        decoder forward
        """
        if update_memory:
            return self.update_memory(
                input_ids=input_ids,
                memory=memory,
                cells=cells,
                state=state,
                use_mem=use_mem
            )

        if memory is not None:
            memory = memory.chunk(2, dim=0)[1]

        outputs = self.decoder.forward(
            input_ids, 
            memory=memory if use_mem else state, 
            labels=labels)

        return outputs


class LlamaARCH14(SegmentRecurrentModifier):
    def __init__(self, model, save_ckp, load_ckp, config):

        self.get_conf(config)
        chunk_size = self.conf["chunk_size"]
        enable_lora = self.conf["enable_lora"]
        lora_kwargs = self.conf["lora_kwargs"]
        tune_mlp = self.conf["tune_mlp"]
        use_fast_attn = self.conf["use_fast_attn"] if "use_fast_attn" in self.conf else False

        encoder = deepcopy(model)
        decoder = model
        eos_token = encoder.model.embed_tokens.weight[2,:]

        encoder = Encoder(
            encoder, 
            chunk_size=chunk_size, 
            tune_mlp=tune_mlp, 
            enable_lora=enable_lora, 
            lora_kwargs=lora_kwargs, 
            use_fast_attn=use_fast_attn)
        
        decoder = Decoder(
            decoder, 
            chunk_size=chunk_size, 
            enable_lora=enable_lora, 
            lora_kwargs=lora_kwargs)

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
            # scores, words = torch.topk(logits, k=20, dim=-1)
            # scores, words = scores.tolist()[0][0], words.tolist()[0][0]

            # generated = tokenizer.decode(input_ids[:,prompt_length:][0])
            # generated = generated.replace('\n', '\\n')
            # print(f"index: {input_ids.shape[-1] - prompt_length}  {generated}")
            # for idx, (score, index) in enumerate(zip(scores, words)):
            #     word = tokenizer.decode(index)
            #     print(f"\t{idx}. {word} {score}".replace('\n', '\\n'))
            # choice = input(">> ")
            # if choice == '':
            #     choice = 0
            # choice = int(choice)
            # new_token = torch.tensor([[words[choice]]])
            # ============================================
            
            # new_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, new_token.to(input_ids.device)], dim=1)
            if new_token.item() in eos_token_id:
                break

            if input_ids.shape[-1] % self.chunk_size == 0:
                past_memory = self.model.update_memory(input_ids=input_ids[:,-self.chunk_size:], memory=past_memory)

        self.reset()
        return input_ids
