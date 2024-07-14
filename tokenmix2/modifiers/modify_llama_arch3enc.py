import torch
import types
from src.modifiers.modify_llama import do_causal_flash_attn, ProjectHead
from peft import get_peft_model, LoraConfig, TaskType


def model_forward(
    self,
    inputs_embeds: torch.Tensor,
    read_memory: torch.Tensor,
    write_memory: torch.Tensor,
    **kwargs
):
    memory = self.model(
        inputs_embeds=inputs_embeds,
        read_memory=read_memory,
        write_memory=write_memory,
    )

    return memory


def model_model_forward(
    self,
    inputs_embeds: torch.Tensor,
    read_memory: torch.Tensor,
    write_memory: torch.Tensor
):
    assert read_memory.ndim == 3 and read_memory.shape[0] == 1 and read_memory.shape[-1] == 4096
    
    residual = read_memory
    hidden_states = inputs_embeds

    before = residual.data.clone()

    for decoder_layer in self.layers:
        hidden_states, read_memory, write_memory = decoder_layer(
            hidden_states=hidden_states,
            read_memory=read_memory,
            write_memory=write_memory
        )

    after = residual.data.clone()
    assert torch.dist(before, after).item() == 0

    residual = residual.to(write_memory.device)
    return write_memory + residual


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    read_memory: torch.Tensor,
    write_memory: torch.Tensor
):
    read_memory = read_memory.to(hidden_states.device)
    write_memory = write_memory.to(hidden_states.device)
    concat_states = torch.cat([read_memory, hidden_states, write_memory], dim=-2)

    residual = concat_states
    concat_states = self.input_layernorm(concat_states)
    concat_states = self.self_attn(concat_states)
    concat_states = residual + concat_states

    residual = concat_states
    concat_states = self.post_attention_layernorm(concat_states)
    concat_states = self.mlp(concat_states)
    concat_states = residual + concat_states

    read_memory, hidden_states, write_memory = concat_states.chunk(3, dim=-2)

    return hidden_states, read_memory, write_memory


def attn_forward(
    self,
    concat_states: torch.Tensor
):
    ques = self.q_proj(concat_states).unflatten(-1, (32,128)).transpose(1,2)
    keys = self.k_proj(concat_states).unflatten(-1, (32,128)).transpose(1,2)
    vals = self.v_proj(concat_states).unflatten(-1, (32,128)).transpose(1,2)

    cos, sin = self.rotary_emb(vals, seq_len=4096)
    attn_output = do_causal_flash_attn(ques, keys, vals, cos, sin, self.o_proj)

    return attn_output


class Encoder(torch.nn.Module):
    def _init_lora(self, lora_rank, lora_alpha, lora_dropout):
        encoder_peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=['q_proj', 'v_proj']
        )
        self.encoder = get_peft_model(self.encoder, encoder_peft_config)


    @property
    def layers(self):
        if self.enable_lora:
            return self.encoder.base_model.model.model.layers
        else:
            return self.encoder.model.layers


    @property
    def model(self):
        if self.enable_lora:
            return self.encoder.base_model.model
        else:
            return self.encoder


    def __init__(self, encoder, chunk_size, enable_lora: bool, lora_kwargs: dict = None):
        super().__init__()
        self.encoder = encoder
        self.chunk_size = chunk_size
        self.enable_lora = False
        
        # 配置新的forward函数
        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)
        for layer in self.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(attn_forward, layer.self_attn)

        # 配置lora
        self.enable_lora = enable_lora
        if self.enable_lora:
            self._init_lora(**lora_kwargs)


    def ft_params(self):
        params = []
        for layer in self.layers:
            if self.enable_lora:
                params += [
                    layer.self_attn.q_proj.lora_A.default.weight,
                    layer.self_attn.q_proj.lora_B.default.weight,
                    layer.self_attn.v_proj.lora_A.default.weight,
                    layer.self_attn.v_proj.lora_B.default.weight
                ]
            else:
                params += [
                    layer.self_attn.q_proj.weight,
                    layer.self_attn.v_proj.weight
                ]
        return params


    def forward(self, input_ids, read_memory, write_memory):
        inputs_embeds = self.model.model.embed_tokens(input_ids).cpu()
        memory = self.encoder(
            inputs_embeds=inputs_embeds, 
            read_memory=read_memory, 
            write_memory=write_memory)
        return memory