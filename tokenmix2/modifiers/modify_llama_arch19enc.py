import torch
import types
from src.modifiers.modify_llama import CrossAttnQKVProj, LlamaRMSNorm, OProj, do_full_flash_attn, do_causal_flash_attn
from src.modifiers.modify_llama_arch19_utils import qkv_proj
from peft import get_peft_model, LoraConfig, TaskType


def model_forward(
    self,
    inputs_embeds: torch.Tensor,
    memory: torch.Tensor,
    **kwargs
):
    memory = self.model(
        inputs_embeds=inputs_embeds,
        memory=memory)

    return memory


def model_model_forward(
    self,
    inputs_embeds: torch.Tensor,
    memory: torch.Tensor
):
    hidden_states = inputs_embeds

    for decoder_layer in self.layers:
        hidden_states = decoder_layer(
            hidden_states=hidden_states,
            memory=memory)

    return hidden_states


def cros_attn_forward(
    qkv_proj: CrossAttnQKVProj,
    out_proj: OProj,
    rotary_emb: torch.nn.Module,
    hidden_states: torch.Tensor,
    memory: torch.Tensor
):
    try:
        ques, keys, vals = qkv_proj(hidden_states, memory)
        cos, sin = rotary_emb(vals, seq_len=4096)
        return do_full_flash_attn(ques, keys, vals, cos, sin, out_proj)
    except:
        import IPython
        IPython.embed(header='debug')


def attn_forward(
    self,
    hidden_states
):
    ques, keys, vals = qkv_proj(hidden_states, self.q_proj, self.k_proj, self.v_proj)
    cos, sin = self.rotary_emb(vals, seq_len=4096)
    return do_causal_flash_attn(ques, keys, vals, cos, sin, self.o_proj)


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    memory: torch.Tensor
):
    # self attention
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states)
    hidden_states = residual + hidden_states

    # cross attention
    if memory is not None:
        memory = memory.to(hidden_states.device)
        residual = hidden_states
        hidden_states = self.cros_attn_norm(hidden_states)
        hidden_states = cros_attn_forward(
            qkv_proj=self.cros_attn_qkv_proj,
            out_proj=self.cros_attn_out_proj,
            rotary_emb=self.self_attn.rotary_emb,
            hidden_states=hidden_states,
            memory=memory)
        hidden_states = residual + hidden_states

    # up projection & down projection
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


class Encoder(torch.nn.Module):
    def _init_lora(
            self, 
            lora_rank: int, 
            lora_alpha: int, 
            lora_dropout: float
        ):
        # self attention
        target_modules = ["q_proj", "v_proj"]
        
        # cross attention
        target_modules += ["que_proj", "key_proj", "val_proj", "out_proj"]

        # mlp
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
            enable_lora: bool, 
            lora_kwargs: dict = None,
        ):

        super().__init__()
        self.encoder = encoder
        self.chunk_size = chunk_size
        self.enable_lora = False

        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)

        for layer in self.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(attn_forward, layer.self_attn)
            layer.cros_attn_qkv_proj = CrossAttnQKVProj(layer)
            layer.cros_attn_out_proj = OProj(layer, zero_init=True)
            layer.cros_attn_norm = LlamaRMSNorm(layer, 4096, eps=1e-6)

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
                    *layer.cros_attn_qkv_proj.get_lora_parameters(),
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
                    *layer.cros_attn_qkv_proj.parameters(),
                    layer.mlp.up_proj.weight,
                    layer.mlp.down_proj.weight,
                    layer.mlp.gate_proj.weight
                ]
            
            params += layer.cros_attn_out_proj.parameters()

        return params


    def forward(
            self, 
            input_ids: torch.Tensor, 
            memory: torch.Tensor
        ):
        inputs_embeds = self.model.model.embed_tokens(input_ids).cpu()
        memory = self.encoder(inputs_embeds=inputs_embeds, memory=memory)
        return memory
