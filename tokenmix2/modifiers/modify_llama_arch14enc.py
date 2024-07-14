import torch
import types
from src.modifiers.modify_llama import ProjectHead, QKVProj
from src.modifiers.modify_llama_arch14_utils import qkv_proj, do_hidden_attn, do_gate_attn, fast_gate_attn
from peft import get_peft_model, LoraConfig, TaskType


def model_forward(
    self,
    inputs_embeds: torch.Tensor,
    cells: torch.Tensor,
    state: torch.Tensor,
    i_gate: torch.Tensor,
    f_gate: torch.Tensor,
    o_gate: torch.Tensor,
    g_gate: torch.Tensor,
    **kwargs
):
    i_gate, f_gate, o_gate, g_gate = self.model(
        inputs_embeds=inputs_embeds,
        state=state,
        i_gate=i_gate,
        f_gate=f_gate,
        o_gate=o_gate,
        g_gate=g_gate)
    
    i_gate = (i_gate + self.i_bias).sigmoid()
    f_gate = (f_gate + self.f_bias).sigmoid()
    o_gate = (o_gate + self.o_bias).sigmoid()
    g_gate = (g_gate + self.g_bias).tanh()

    cells = i_gate * g_gate + f_gate * cells
    state = o_gate * cells.tanh()

    return cells, state


def model_model_forward(
    self,
    inputs_embeds: torch.Tensor,
    state: torch.Tensor,
    i_gate: torch.Tensor,
    f_gate: torch.Tensor,
    o_gate: torch.Tensor,
    g_gate: torch.Tensor
):
    hidden_states = inputs_embeds
    i_gate_states = i_gate
    f_gate_states = f_gate
    o_gate_states = o_gate
    g_gate_states = g_gate

    states_records = []

    for decoder_layer, memory_states in zip(self.layers, state.chunk(32, dim=0)):
        
        states_records.append([
            i_gate_states.cpu(),
            f_gate_states.cpu(),
            o_gate_states.cpu(),
            g_gate_states.cpu()
        ])
        
        hidden_states, i_gate_states, f_gate_states, o_gate_states, g_gate_states = decoder_layer(
            hidden_states=hidden_states,
            memory_states=memory_states,
            i_gate_states=i_gate_states,
            f_gate_states=f_gate_states,
            o_gate_states=o_gate_states,
            g_gate_states=g_gate_states)
    
    i_gate = torch.cat([gates[0] for gates in states_records])
    f_gate = torch.cat([gates[1] for gates in states_records])
    o_gate = torch.cat([gates[2] for gates in states_records])
    g_gate = torch.cat([gates[3] for gates in states_records])
    
    return i_gate, f_gate, o_gate, g_gate


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    memory_states: torch.Tensor,
    i_gate_states: torch.Tensor,
    f_gate_states: torch.Tensor,
    o_gate_states: torch.Tensor,
    g_gate_states: torch.Tensor
):
    memory_states = memory_states.to(hidden_states.device)
    i_gate_states = i_gate_states.to(hidden_states.device)
    f_gate_states = f_gate_states.to(hidden_states.device)
    o_gate_states = o_gate_states.to(hidden_states.device)
    g_gate_states = g_gate_states.to(hidden_states.device)

    concat_states = torch.cat([
        hidden_states, 
        i_gate_states, 
        f_gate_states,
        o_gate_states,
        g_gate_states], 
        dim=-2)

    residual = concat_states
    concat_states = self.input_layernorm(concat_states)
    concat_states = self.self_attn(memory_states, *concat_states.chunk(5, dim=-2))
    concat_states = residual + concat_states

    residual = concat_states
    concat_states = self.post_attention_layernorm(concat_states)
    concat_states = self.mlp(concat_states)
    concat_states = residual + concat_states

    return concat_states.chunk(5, dim=-2)


def attn_forward(
    self,
    memory_states: torch.Tensor,
    hidden_states: torch.Tensor,
    i_gate_states: torch.Tensor,
    f_gate_states: torch.Tensor,
    o_gate_states: torch.Tensor,
    g_gate_states: torch.Tensor
):
    mem_keys, mem_vals = self.project_head(memory_states)

    # NOTE: step 1. compute the corresponding output of `hidden_states`
    hidden_ques, hidden_keys, hidden_vals = qkv_proj(
        hidden_states, 
        self.q_proj, 
        self.k_proj, 
        self.v_proj)
    hidden_keys = torch.cat([mem_keys, hidden_keys], dim=-2)
    hidden_vals = torch.cat([mem_vals, hidden_vals], dim=-2)
    cos, sin = self.rotary_emb(hidden_vals, seq_len=4096)

    hidden_outs = do_hidden_attn(
        hidden_ques, 
        hidden_keys, 
        hidden_vals, 
        cos, 
        sin,
        self.o_proj)

    # NOTE: step 2. compute the corresponding output of `i_gate_states`
    i_gate_ques, i_gate_keys, i_gate_vals = self.i_gate_proj(i_gate_states)
    i_gate_keys = torch.cat([hidden_keys, i_gate_keys], dim=-2)
    i_gate_vals = torch.cat([hidden_vals, i_gate_vals], dim=-2)

    if self.use_fast_attn:
        i_gate_outs = fast_gate_attn(
            i_gate_ques,
            i_gate_keys,
            i_gate_vals,
            cos,
            sin,
            self.layer_idx,
            self.o_proj)
    else:
        i_gate_outs = do_gate_attn(
            query=i_gate_ques, 
            key=i_gate_keys, 
            value=i_gate_vals, 
            cos=cos, 
            sin=sin, 
            layer_id=self.layer_idx, 
            o_proj=self.o_proj)
        
    # NOTE: step 3. compute the corresponding output of `f_gate_states`
    f_gate_ques, f_gate_keys, f_gate_vals = self.f_gate_proj(f_gate_states)
    f_gate_keys = torch.cat([hidden_keys, f_gate_keys], dim=-2)
    f_gate_vals = torch.cat([hidden_vals, f_gate_vals], dim=-2)

    if self.use_fast_attn:
        f_gate_outs = fast_gate_attn(
            f_gate_ques,
            f_gate_keys,
            f_gate_vals,
            cos,
            sin,
            self.layer_idx,
            self.o_proj
        )
    else:
        f_gate_outs = do_gate_attn(
            query=f_gate_ques, 
            key=f_gate_keys, 
            value=f_gate_vals, 
            cos=cos, 
            sin=sin, 
            layer_id=self.layer_idx, 
            o_proj=self.o_proj)
        
    # NOTE: step 4. compute the corresponding output of `o_gate_states`
    o_gate_ques, o_gate_keys, o_gate_vals = self.o_gate_proj(o_gate_states)
    o_gate_keys = torch.cat([hidden_keys, o_gate_keys], dim=-2)
    o_gate_vals = torch.cat([hidden_vals, o_gate_vals], dim=-2)

    if self.use_fast_attn:
        o_gate_outs = fast_gate_attn(
            o_gate_ques,
            o_gate_keys,
            o_gate_vals,
            cos,
            sin,
            self.layer_idx,
            self.o_proj
        )
    else:
        o_gate_outs = do_gate_attn(
            query=o_gate_ques, 
            key=o_gate_keys, 
            value=o_gate_vals, 
            cos=cos, 
            sin=sin, 
            layer_id=self.layer_idx, 
            o_proj=self.o_proj)
    
    # NOTE: step 5. compute the corresponding output of `g_gate_states`
    g_gate_ques, g_gate_keys, g_gate_vals = self.g_gate_proj(g_gate_states)
    g_gate_keys = torch.cat([hidden_keys, g_gate_keys], dim=-2)
    g_gate_vals = torch.cat([hidden_vals, g_gate_vals], dim=-2)

    if self.use_fast_attn:
        g_gate_outs = fast_gate_attn(
            g_gate_ques,
            g_gate_keys,
            g_gate_vals,
            cos,
            sin,
            self.layer_idx,
            self.o_proj
        )
    else:
        g_gate_outs = do_gate_attn(
            query=g_gate_ques, 
            key=g_gate_keys, 
            value=g_gate_vals, 
            cos=cos, 
            sin=sin, 
            layer_id=self.layer_idx, 
            o_proj=self.o_proj)

    concat_outs = torch.cat([
        hidden_outs,
        i_gate_outs,
        f_gate_outs,
        o_gate_outs,
        g_gate_outs
    ], dim=-2)
    
    return concat_outs


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
        self.model.i_bias = torch.nn.Parameter(torch.zeros((32, chunk_size, 4096), dtype=torch.bfloat16), requires_grad=True)
        self.model.f_bias = torch.nn.Parameter(torch.ones((32, chunk_size, 4096), dtype=torch.bfloat16), requires_grad=True)
        self.model.o_bias = torch.nn.Parameter(torch.zeros((32, chunk_size, 4096), dtype=torch.bfloat16), requires_grad=True)
        self.model.g_bias = torch.nn.Parameter(torch.zeros((32, chunk_size, 4096), dtype=torch.bfloat16), requires_grad=True)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)

        for layer in self.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(attn_forward, layer.self_attn)
            layer.self_attn.project_head = ProjectHead(layer)
            layer.self_attn.i_gate_proj = QKVProj(layer)
            layer.self_attn.f_gate_proj = QKVProj(layer)
            layer.self_attn.o_gate_proj = QKVProj(layer)
            layer.self_attn.g_gate_proj = QKVProj(layer)
            layer.self_attn.use_fast_attn = use_fast_attn

        self.enable_lora = enable_lora
        if self.enable_lora:
            self._init_lora(**lora_kwargs)


    def ft_params(self):
        params = [self.model.i_bias, self.model.f_bias, self.model.o_bias, self.model.g_bias]

        for layer in self.layers:
            if self.enable_lora:
                params += [
                    layer.self_attn.q_proj.lora_A.default.weight,
                    layer.self_attn.q_proj.lora_B.default.weight,
                    layer.self_attn.v_proj.lora_A.default.weight,
                    layer.self_attn.v_proj.lora_B.default.weight,
                    *layer.self_attn.project_head.get_lora_parameters(),
                    *layer.self_attn.i_gate_proj.get_lora_parameters(),
                    *layer.self_attn.f_gate_proj.get_lora_parameters(),
                    *layer.self_attn.o_gate_proj.get_lora_parameters(),
                    *layer.self_attn.g_gate_proj.get_lora_parameters()
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
                    *layer.self_attn.i_gate_proj.parameters(),
                    *layer.self_attn.f_gate_proj.parameters(),
                    *layer.self_attn.o_gate_proj.parameters(),
                    *layer.self_attn.g_gate_proj.parameters()
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
            cells: torch.Tensor,
            state: torch.Tensor,
            i_gate: torch.Tensor,
            f_gate: torch.Tensor,
            o_gate: torch.Tensor,
            g_gate: torch.Tensor
        ):

        assert cells.ndim == 3 and cells.shape[0] == 32
        assert state.ndim == 3 and state.shape[0] == 32

        inputs_embeds = self.model.model.embed_tokens(input_ids).cpu()
        cells, state = self.encoder(
            inputs_embeds=inputs_embeds,
            cells=cells,
            state=state,
            i_gate=i_gate,
            f_gate=f_gate,
            o_gate=o_gate,
            g_gate=g_gate
        )

        assert cells.ndim == 3 and cells.shape[0] == 32
        assert state.ndim == 3 and state.shape[0] == 32

        return cells, state
