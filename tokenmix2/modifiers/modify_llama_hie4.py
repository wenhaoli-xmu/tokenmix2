import torch
import torch.utils.checkpoint
from src.modifier import Modifier, SegmentRecurrentModifier
from copy import deepcopy

from src.modifiers.modify_llama_hie4enc import Encoder
from src.modifiers.modify_llama_hie4dec import Decoder


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, chunk_size, eos_token):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.chunk_size = chunk_size
        self.init_memory = torch.nn.Parameter(eos_token[None,None,:].expand(32,self.chunk_size,-1).detach(), requires_grad=True)

    def ft_params(self):
        return self.encoder.ft_params() + self.decoder.ft_params() + [self.init_memory]

    def reset(self):
        ...

    def update_memory(self, input_ids, memory=None, **kwargs):

        # NOTE：当input_ids长度达标时，使用encoder对memory进行更新得到updated_memory
        updated_memory = self.encoder(
            input_ids=input_ids, 
            memory=memory if memory is not None else self.init_memory
        ) if input_ids.shape[-1] == self.chunk_size else None

        return updated_memory
        
    def forward(self, input_ids, memory=None, labels=None):

        assert input_ids.shape[1] <= self.chunk_size
        assert memory is None or memory.shape == (32, self.chunk_size, 4096)

        outputs = self.decoder.forward(input_ids, memory=memory, labels=labels)

        return outputs


class LlamaHIE4(SegmentRecurrentModifier):
    def __init__(self, model, save_ckp, load_ckp, config):

        self.get_conf(config)
        chunk_size = self.conf["chunk_size"]
        enable_lora = self.conf["enable_lora"]
        lora_kwargs = self.conf["lora_kwargs"]

        encoder = deepcopy(model)
        decoder = model
        eos_token = encoder.model.embed_tokens.weight[2,:]

        encoder = Encoder(encoder, chunk_size=chunk_size, enable_lora=enable_lora, lora_kwargs=lora_kwargs)
        decoder = Decoder(decoder, chunk_size=chunk_size, enable_lora=enable_lora, lora_kwargs=lora_kwargs)
        encoder_decoder = Model(encoder, decoder, chunk_size=chunk_size, eos_token=eos_token)

        super().__init__(encoder_decoder, save_ckp, load_ckp, chunk_size=chunk_size)

    def ft_params(self):
        return self.model.ft_params()

    def reset(self):
        self.model.reset()

    def get_memories(self, segment_id):
        ...
