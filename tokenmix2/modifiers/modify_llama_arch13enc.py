import torch
import types
from ..modifiers.modify_llama import ProjectHead, QKVProj
from ..modifiers.modify_llama_arch13_utils import qkv_proj, do_hidden_attn, do_gate_attn, fast_gate_attn
from peft import get_peft_model, LoraConfig, TaskType


def model_forward(
    self,
    inputs_embeds: torch.Tensor,
    memory: torch.Tensor,
    beacon: torch.Tensor,
    forget: torch.Tensor,
    **kwargs
):
    memory = self.model(
        inputs_embeds=inputs_embeds,
        memory=memory,
        beacon=beacon,
        forget=forget)
    
    return memory


def model_model_forward(
    self,
    inputs_embeds: torch.Tensor,
    memory: torch.Tensor,
    beacon: torch.Tensor,
    forget: torch.Tensor
):
    hidden_states = inputs_embeds
    beacon_states = beacon
    forget_states = forget

    states_records = []

    for decoder_layer, memory_states in zip(self.layers, memory.chunk(32, dim=0)):
        states_records.append([beacon_states.cpu(), forget_states.cpu()])
        hidden_states, beacon_states, forget_states = decoder_layer(
            hidden_states=hidden_states,
            memory_states=memory_states,
            beacon_states=beacon_states,
            forget_states=forget_states)
    
    inject = torch.cat([state[0] for state in states_records], dim=0)
    forget = torch.cat([state[1] for state in states_records], dim=0)
    memory = memory * forget.sigmoid() + inject * (1 - forget.sigmoid())

    return memory


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    memory_states: torch.Tensor,
    beacon_states: torch.Tensor,
    forget_states: torch.Tensor
):
    memory_states = memory_states.to(hidden_states.device)
    beacon_states = beacon_states.to(hidden_states.device)
    forget_states = forget_states.to(hidden_states.device)
    concat_states = torch.cat([hidden_states, beacon_states, forget_states], dim=-2)

    # self attention
    residual = concat_states
    concat_states = self.input_layernorm(concat_states)
    concat_states = self.self_attn(memory_states, *concat_states.chunk(3, dim=-2))
    concat_states = residual + concat_states

    residual = concat_states
    concat_states = self.post_attention_layernorm(concat_states)
    concat_states = self.mlp(concat_states)
    concat_states = residual + concat_states

    hidden_states, beacon_states, forget_states = concat_states.chunk(3, dim=-2)

    return hidden_states, beacon_states, forget_states


def attn_forward(
    self,
    memory_states: torch.Tensor,
    hidden_states: torch.Tensor,
    beacon_states: torch.Tensor,
    forget_states: torch.Tensor
):
    mem_keys, mem_vals = self.project_head(memory_states)

    # NOTE: step 1. compute the corresponding output of `hidden_states`
    hid_ques, hid_keys, hid_vals = qkv_proj(
        hidden_states, 
        self.q_proj, 
        self.k_proj, 
        self.v_proj)
    hid_keys = torch.cat([mem_keys, hid_keys], dim=-2)
    hid_vals = torch.cat([mem_vals, hid_vals], dim=-2)
    cos, sin = self.rotary_emb(hid_vals, seq_len=4096)

    hid_outs = do_hidden_attn(
        hid_ques, 
        hid_keys, 
        hid_vals, 
        cos, 
        sin,
        self.o_proj)

    # NOTE: step 2. compute the corresponding output of `beacon_states`
    bcn_ques, bcn_keys, bcn_vals = self.bcn_proj(beacon_states)
    bcn_keys = torch.cat([hid_keys, bcn_keys], dim=-2)
    bcn_vals = torch.cat([hid_vals, bcn_vals], dim=-2)

    if self.use_fast_attn:
        bcn_outs = fast_gate_attn(
            bcn_ques,
            bcn_keys,
            bcn_vals,
            cos,
            sin,
            self.layer_idx,
            self.o_proj)
    else:
        bcn_outs = do_gate_attn(
            bcn_ques, 
            bcn_keys, 
            bcn_vals, 
            cos, 
            sin, 
            self.layer_idx, 
            self.o_proj)

    # NOTE: step 3. compute the output regard to `forget_states`
    fgt_ques, fgt_keys, fgt_vals = self.fgt_proj(forget_states)
    fgt_keys = torch.cat([hid_keys, fgt_keys], dim=-2)
    fgt_vals = torch.cat([hid_vals, fgt_vals], dim=-2)

    if self.use_fast_attn:
        fgt_outs = fast_gate_attn(
            fgt_ques,
            fgt_keys,
            fgt_vals,
            cos,
            sin,
            self.layer_idx,
            self.o_proj)
    else:
        fgt_outs = do_gate_attn(
            fgt_ques,
            fgt_keys,
            fgt_vals,
            cos,
            sin,
            self.layer_idx,
            self.o_proj)
    
    return torch.cat([hid_outs, bcn_outs, fgt_outs], dim=-2)


class Encoder(torch.nn.Module):
    def _init_lora(
            self, 
            lora_rank: int, 
            lora_alpha: int, 
            lora_dropout: float
        ):
        target_modules = ["q_proj", "v_proj", "que_proj", "key_proj", "val_proj"]
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
            tune_mlp: bool,
            enable_lora: bool, 
            lora_kwargs: dict = None,
            use_fast_attn: bool = False
        ):

        super().__init__()
        self.encoder = encoder
        self.chunk_size = chunk_size
        self.tune_mlp = tune_mlp
        self.enable_lora = False

        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)

        for layer in self.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(attn_forward, layer.self_attn)
            layer.self_attn.project_head = ProjectHead(layer)
            layer.self_attn.bcn_proj = QKVProj(layer)
            layer.self_attn.fgt_proj = QKVProj(layer)
            layer.self_attn.use_fast_attn = use_fast_attn

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
                    *layer.self_attn.project_head.get_lora_parameters(),
                    *layer.self_attn.bcn_proj.get_lora_parameters(),
                    *layer.self_attn.fgt_proj.get_lora_parameters()
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
                    *layer.self_attn.project_head.parameters(),
                    *layer.self_attn.bcn_proj.parameters(),
                    *layer.self_attn.fgt_proj.parameters()
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
            beacon: torch.Tensor,
            forget: torch.Tensor
        ):
        inputs_embeds = self.model.model.embed_tokens(input_ids).cpu()
        memory = self.encoder(inputs_embeds=inputs_embeds, memory=memory, beacon=beacon, forget=forget)
        return memory
