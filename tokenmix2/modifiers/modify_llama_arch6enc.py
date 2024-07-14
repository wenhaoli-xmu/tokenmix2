import torch
import types
from src.modifiers.modify_llama import do_beacon_attn, ProjectHead
from peft import get_peft_model, LoraConfig, TaskType

from typing import List


def model_forward(
    self,
    inputs_embeds: torch.Tensor,
    memory: List[torch.Tensor],
    beacon: List[torch.Tensor],
    **kwargs
):
    memory = self.model(
        inputs_embeds=inputs_embeds,
        memory=memory,
        beacon=beacon)
    return memory


def model_model_forward(
    self,
    inputs_embeds: torch.Tensor,
    memory: torch.Tensor,
    beacon: torch.Tensor
):
    hidden_states = inputs_embeds
    beacon_states = beacon
    update_states = []

    for decoder_layer, memory_states in zip(self.layers, memory.chunk(32, dim=0)):
        update_states.append(beacon_states.cpu())
        hidden_states, beacon_states = decoder_layer(
            hidden_states=hidden_states,
            memory_states=memory_states,
            beacon_states=beacon_states)

    return torch.cat(update_states, dim=0)


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    memory_states: torch.Tensor,
    beacon_states: torch.Tensor
):
    memory_states = memory_states.to(hidden_states.device)
    beacon_states = beacon_states.to(hidden_states.device)
    concat_states = torch.cat([hidden_states, beacon_states], dim=-2)

    # self attention
    residual = concat_states
    concat_states = self.input_layernorm(concat_states)
    concat_states = self.self_attn(memory_states, concat_states)
    concat_states = residual + concat_states

    residual = concat_states
    concat_states = self.post_attention_layernorm(concat_states)
    concat_states = self.mlp(concat_states)
    concat_states = residual + concat_states

    hidden_states, beacon_states = concat_states.chunk(2, dim=-2)

    return hidden_states, beacon_states


def attn_forward(
    self,
    memory_states: torch.Tensor,
    concat_states: torch.Tensor
):
    ques = self.q_proj(concat_states).unflatten(-1, (32,128)).transpose(1,2)
    keys = self.k_proj(concat_states).unflatten(-1, (32,128)).transpose(1,2)
    vals = self.v_proj(concat_states).unflatten(-1, (32,128)).transpose(1,2)

    mem_keys, mem_vals = self.project_head(memory_states)
    keys = torch.cat([mem_keys, keys], dim=-2)
    vals = torch.cat([mem_vals, vals], dim=-2)

    cos, sin = self.rotary_emb(vals, seq_len=4096)
    attn_output = do_beacon_attn(
        query=ques, 
        key=keys, 
        value=vals, 
        cos=cos, 
        sin=sin, 
        o_proj=self.o_proj, 
        num_ordinal=64, 
        num_memory=64, 
        num_beacons=64, 
        layer_id=self.layer_idx,
        memory_mask=self.memory_mask)

    return attn_output


class Encoder(torch.nn.Module):
    def _init_lora(
            self, 
            lora_rank: int, 
            lora_alpha: int, 
            lora_dropout: float
        ):
        target_modules = ["q_proj", "v_proj", "key_proj", "val_proj"]
        if self.tune_mlp:
            target_modules += ["up_proj", "down_proj", "gate_proj"]

        encoder_peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules
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


    def __init__(
            self, 
            encoder, 
            chunk_size,
            memory_mask: str,
            tune_mlp: bool,
            enable_lora: bool, 
            lora_kwargs: dict = None
        ):

        super().__init__()
        self.encoder = encoder
        self.chunk_size = chunk_size
        self.memory_mask = memory_mask
        self.tune_mlp = tune_mlp
        self.enable_lora = False

        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)
        for layer in self.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(attn_forward, layer.self_attn)
            layer.self_attn.project_head = ProjectHead(layer)
            layer.self_attn.memory_mask = memory_mask

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
                    layer.self_attn.v_proj.lora_B.default.weight,
                    *layer.self_attn.project_head.get_lora_parameters()
                ]
                if self.tune_mlp:
                    params += [
                        layer.mlp.up_proj.lora_A.default.weight,
                        layer.mlp.up_proj.lora_B.default.weight,
                        layer.mlp.down_proj.lora_A.default.weight,
                        layer.mlp.down_proj.lora_B.default.weight,
                        layer.mlp.gate_proj.lora_A.default.weight,
                        layer.mlp.gate_proj.lora_B.default.weight
                    ]
            else:
                params += [
                    layer.self_attn.q_proj.weight,
                    layer.self_attn.k_proj.weight,
                    layer.self_attn.v_proj.weight,
                    layer.self_attn.o_proj.weight,
                    *layer.self_attn.project_head.parameters()
                ]
                if self.tune_mlp:
                    params += [
                        layer.mlp.up_proj.weight,
                        layer.mlp.down_proj.weight,
                        layer.mlp.gate_proj.weight
                    ]
        return params


    def forward(
            self, 
            input_ids: torch.Tensor, 
            memory: torch.Tensor, 
            beacon: torch.Tensor
        ):
        inputs_embeds = self.model.model.embed_tokens(input_ids).cpu()
        memory = self.encoder(inputs_embeds=inputs_embeds, memory=memory, beacon=beacon)
        return memory