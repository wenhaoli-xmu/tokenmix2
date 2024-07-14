import torch
import torch.utils.checkpoint
from src.modifier import SegmentRecurrentModifier
from copy import deepcopy
from typing import List
from functools import partial

from src.modifiers.modify_llama import segment
from src.modifiers.modify_llama_arch17enc import Encoder
from src.modifiers.modify_llama_arch17dec import Decoder


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
        self.init_memory = torch.zeros_like(eos_token)[None,None,:].expand(32,chunk_size,-1)
        self.beacon = torch.nn.Parameter(torch.zeros_like(eos_token)[None,None,:].expand(-1,chunk_size,-1).clone(), requires_grad=True)
        self.forget = torch.nn.Parameter(torch.zeros_like(eos_token)[None,None,:].expand(-1,chunk_size,-1).clone(), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.beacon.data)
        torch.nn.init.xavier_uniform_(self.forget.data)

    def ft_params(self):
        params = self.encoder.ft_params() + self.decoder.ft_params()
        params += [self.beacon, self.forget]
        return params

    def reset(self):
        self.decoder.reset()

    def update_memory(
            self, 
            input_ids, 
            memory=None, 
            do_not_compress: bool = False,
            **kwargs
        ):
        if do_not_compress:
            return memory

        """
        encoder forward
        """
        updated_memory = self.encoder(
            input_ids=input_ids, 
            memory=memory if memory is not None else self.init_memory,
            beacon=self.beacon,
            forget=self.forget
        ) if input_ids.shape[-1] == self.chunk_size else None

        self.decoder.reset()

        return updated_memory
        
    def forward(
            self, 
            input_ids, 
            memory=None, 
            labels=None,
            clear_cache: int = None,
            **kwargs
        ):

        """
        decoder forward
        """
        outputs = self.decoder.forward(input_ids, memory=memory, labels=labels)
        if clear_cache is not None:
            self.decoder.clear_kv_cache(clear_cache)

        return outputs


class LlamaARCH17(SegmentRecurrentModifier):
    def __init__(self, model, save_ckp, load_ckp, config):

        self.get_conf(config)
        chunk_size = self.conf["chunk_size"]
        enable_lora = self.conf["enable_lora"]
        lora_kwargs = self.conf["lora_kwargs"]
        tune_mlp = self.conf["tune_mlp"]
        use_fast_attn = self.conf["use_fast_attn"] if "use_fast_attn" in self.conf else False
        tune_decoder = self.conf["tune_decoder"] if "tune_decoder" in self.conf else False

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
