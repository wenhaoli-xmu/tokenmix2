import torch
import types
from src.modifiers.modify_llama import ProjectHead, QKVProj
from src.modifiers.modify_llama_arch11_utils import do_hidden_attn, do_gate_attn, qkv_proj
from peft import get_peft_model, LoraConfig, TaskType


def model_forward(
    self,
    inputs_embeds: torch.Tensor,
    cells: torch.Tensor,  # 32 layers
    state: torch.Tensor,  # 32 layers
    i_gate: torch.Tensor, # 1 layers
    f_gate: torch.Tensor, 
    o_gate: torch.Tensor,
    g_gate: torch.Tensor,
    **kwargs
):
    """
    cells, state
    ------------
    both have 32 layers

    i_gate, f_gate, o_gate, g_gate
    ------------------------------
    only have 1 layer
    """
    i_gate, f_gate, o_gate, g_gate = self.model(
        inputs_embeds=inputs_embeds,
        state=state,
        i_gate=i_gate,
        f_gate=f_gate,
        o_gate=o_gate,
        g_gate=g_gate)
    
    i_gate = i_gate.sigmoid()
    f_gate = f_gate.sigmoid()
    o_gate = o_gate.sigmoid()
    g_gate = g_gate.tanh()

    cells = i_gate * g_gate + f_gate * cells
    state = o_gate * cells.tanh()

    """
    cells, state
    ------------
    32x layers
    """
    
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
    memory_keys, memory_vals = self.project_head(memory_states)

    # 1. compute outputs with regard to hidden_states
    hidden_ques, hidden_keys, hidden_vals = qkv_proj(
        hidden_states, 
        self.q_proj, 
        self.k_proj, 
        self.v_proj)
    hidden_keys = torch.cat([memory_keys, hidden_keys], dim=-2)
    hidden_vals = torch.cat([memory_vals, hidden_vals], dim=-2)
    cos, sin = self.rotary_emb(hidden_vals, seq_len=4096)

    hidden_outs = do_hidden_attn(
        query=hidden_ques, 
        key=hidden_keys, 
        value=hidden_vals, 
        cos=cos, 
        sin=sin, 
        out_proj=self.o_proj)

    # 2. compute outputs with regard to i_gate_states
    i_gate_ques, i_gate_keys, i_gate_vals = self.i_gate_proj(i_gate_states)
    i_gate_keys = torch.cat([hidden_keys, i_gate_keys], dim=-2)
    i_gate_vals = torch.cat([hidden_vals, i_gate_vals], dim=-2)

    i_gate_outs = do_gate_attn(
        query=i_gate_ques, 
        key=i_gate_keys, 
        value=i_gate_vals, 
        cos=cos, 
        sin=sin, 
        layer_id=self.layer_idx, 
        o_proj=self.o_proj)

    # 3. compute outputs with regard to f_gate_states
    f_gate_ques, f_gate_keys, f_gate_vals = self.f_gate_proj(f_gate_states)
    f_gate_keys = torch.cat([hidden_keys, f_gate_keys], dim=-2)
    f_gate_vals = torch.cat([hidden_vals, f_gate_vals], dim=-2)

    f_gate_outs = do_gate_attn(
        query=f_gate_ques, 
        key=f_gate_keys, 
        value=f_gate_vals, 
        cos=cos, 
        sin=sin, 
        layer_id=self.layer_idx, 
        o_proj=self.o_proj)
    
    # 4. compute outputs with regard to o_gate_states
    o_gate_ques, o_gate_keys, o_gate_vals = self.o_gate_proj(o_gate_states)
    o_gate_keys = torch.cat([hidden_keys, o_gate_keys], dim=-2)
    o_gate_vals = torch.cat([hidden_vals, o_gate_vals], dim=-2)

    o_gate_outs = do_gate_attn(
        query=o_gate_ques, 
        key=o_gate_keys, 
        value=o_gate_vals, 
        cos=cos, 
        sin=sin, 
        layer_id=self.layer_idx, 
        o_proj=self.o_proj)
    
    # 5. compute outputs with regard to g_gate_states
    g_gate_ques, g_gate_keys, g_gate_vals = self.g_gate_proj(g_gate_states)
    g_gate_keys = torch.cat([hidden_keys, g_gate_keys], dim=-2)
    g_gate_vals = torch.cat([hidden_vals, g_gate_vals], dim=-2)

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
        g_gate_outs], dim=-2)
    
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
            gain: float = 1
        ):

        super().__init__()
        self.encoder = encoder
        self.chunk_size = chunk_size
        self.tune_mlp = tune_mlp
        self.enable_lora = False

        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)
        self.model.model.gain = gain
        for layer in self.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(attn_forward, layer.self_attn)
            layer.self_attn.project_head = ProjectHead(layer)
            layer.self_attn.i_gate_proj = QKVProj(layer)
            layer.self_attn.f_gate_proj = QKVProj(layer)
            layer.self_attn.o_gate_proj = QKVProj(layer)
            layer.self_attn.g_gate_proj = QKVProj(layer)

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
            g_gate=g_gate)
        
        assert cells.ndim == 3 and cells.shape[0] == 32
        assert state.ndim == 3 and state.shape[0] == 32
        return cells, state