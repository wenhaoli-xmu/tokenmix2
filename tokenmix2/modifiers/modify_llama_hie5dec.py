import torch
import types
from src.modifiers.modify_llama import compute_loss, do_causal_attn, check_and_apply_rope, CrossAttnQKVProj, OProj, do_causal_flash_attn
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast, repeat_kv, CrossEntropyLoss
from torch.utils.checkpoint import checkpoint

from peft import get_peft_model, LoraConfig, TaskType
from typing import List


def model_forward(
    self,
    input_ids: torch.LongTensor,
    labels: torch.Tensor = None,
    memory: torch.Tensor = None,
    prefill: bool = False,
    **kwargs
):
    if memory is not None:
        assert memory.ndim == 4
    
    rets = self.model(
        input_ids=input_ids,
        memory=memory,
        prefill=prefill)

    if prefill:
        return rets
    else:
        hidden_states = rets

    logits = self.lm_head(hidden_states).float()

    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)

        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    return CausalLMOutputWithPast(loss=loss, logits=logits)


def model_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    memory: torch.Tensor = None,
    prefill: bool = False
):
    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    if memory is None:
        memory = [None] * len(self.layers)
    assert len(memory) == len(self.layers)

    if prefill:
        activations = []

    for decoder_layer, mem in zip(self.layers, memory):

        if prefill:
            activations.append(hidden_states)

        hidden_states = checkpoint(
            decoder_layer,
            hidden_states,
            mem,
            prefill,
            use_reentrant=False)

    hidden_states = self.norm(hidden_states)

    if prefill:
        activations = [act.cpu() for act in activations]
        mems = torch.stack(activations, dim=0)
        assert mems.ndim == 4
        return mems
    else:
        return hidden_states


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    memory: torch.Tensor = None,
    prefill: bool = False,
):
    # self attention
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states)
    hidden_states = residual + hidden_states

    # cross attention
    if not prefill and memory is not None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        for mem in memory.chunk(chunks=memory.shape[0], dim=0):
            hidden_states = cross_attn(
                config=self.self_attn.config,
                qkv_proj=self.cros_attn_qkv_proj,
                out_proj=self.cros_attn_out_proj,
                rotary_emb=self.self_attn.rotary_emb,
                hidden_states=hidden_states,
                memory=mem)
        hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


def cross_attn(
        config,
        qkv_proj: CrossAttnQKVProj,
        out_proj: OProj,
        rotary_emb: torch.nn.Module,
        hidden_states: torch.Tensor,
        memory: torch.Tensor
):
    memory = memory.to(hidden_states.device)

    num_heads, embed_dim = config.num_attention_heads, config.hidden_size
    head_dim = embed_dim // num_heads
    max_pos_embed = config.max_position_embeddings
    num_kv_heads = config.num_key_value_heads
    num_kv_group = config.num_attention_heads // num_kv_heads

    ques, keys, vals = qkv_proj(hidden_states, memory, num_query_head=num_heads, num_kv_head=num_kv_heads, head_dim=head_dim)
    cos, sin = rotary_emb(vals, seq_len=max_pos_embed)
    keys = repeat_kv(keys, num_kv_group)
    vals = repeat_kv(vals, num_kv_group)

    return do_causal_flash_attn(ques, keys, vals, cos, sin, out_proj)
    


def self_attn_forward(self, hidden_states: torch.Tensor):
    num_heads, embed_dim = self.config.num_attention_heads, self.config.hidden_size
    head_dim = embed_dim // num_heads
    max_pos_embed = self.config.max_position_embeddings
    num_kv_heads = self.config.num_key_value_heads
    num_kv_group = self.config.num_attention_heads // num_kv_heads

    ques = self.q_proj(hidden_states).unflatten(-1, (num_heads,head_dim)).transpose(1,2)
    keys = self.k_proj(hidden_states).unflatten(-1, (num_kv_heads,head_dim)).transpose(1,2)
    vals = self.v_proj(hidden_states).unflatten(-1, (num_kv_heads,head_dim)).transpose(1,2)
    keys = repeat_kv(keys, num_kv_group)
    vals = repeat_kv(vals, num_kv_group)

    cos, sin = self.rotary_emb(vals, seq_len=max_pos_embed)

    return do_causal_flash_attn(
        query=ques,
        key=keys,
        value=vals,
        cos=cos,
        sin=sin,
        out_proj=self.o_proj)


class Decoder(torch.nn.Module):
    def _init_lora(
            self,
            lora_rank: int, 
            lora_alpha: int, 
            lora_dropout: float
        ):
        target_modules = r".*\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj"
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules
        )
        self.decoder = get_peft_model(self.decoder, peft_config)


    @property
    def layers(self):
        if self.enable_lora:
            return self.decoder.base_model.model.model.layers
        else:
            return self.decoder.model.layers


    @property
    def model(self):
        if self.enable_lora:
            return self.decoder.base_model.model
        else:
            return self.decoder


    def reset(self):
        for layer in self.layers:
            if hasattr(layer.self_attn, 'decoding_k_cache'):
                del layer.self_attn.decoding_k_cache
                del layer.self_attn.decoding_v_cache


    def __init__(
            self, 
            decoder, 
            chunk_size,
            enable_lora: bool = False,
            lora_kwargs: dict = None,
        ):
        super().__init__()
        self.decoder = decoder
        self.chunk_size = chunk_size
        self.enable_lora = False
        embed_dim = self.decoder.config.hidden_size

        # 修改各种forward函数
        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)
        for layer in self.layers:
            layer.cros_attn_qkv_proj = CrossAttnQKVProj(layer, random_init=False, embed_dim=embed_dim)
            layer.cros_attn_out_proj = OProj(layer, zero_init=True, embed_dim=embed_dim)
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)

        self.enable_lora = enable_lora
        if self.enable_lora is True:
            self._init_lora(**lora_kwargs)

    def ft_params(self):
        params = []
        for layer in self.layers:
            if self.enable_lora:
                params += [
                    layer.self_attn.q_proj.lora_A.default.weight,
                    layer.self_attn.q_proj.lora_B.default.weight,
                    layer.self_attn.k_proj.lora_A.default.weight,
                    layer.self_attn.k_proj.lora_B.default.weight,
                    layer.self_attn.v_proj.lora_A.default.weight,
                    layer.self_attn.v_proj.lora_B.default.weight,
                    layer.self_attn.o_proj.lora_A.default.weight,
                    layer.self_attn.o_proj.lora_B.default.weight,
                    layer.mlp.gate_proj.lora_A.default.weight,
                    layer.mlp.gate_proj.lora_B.default.weight,
                    layer.mlp.up_proj.lora_A.default.weight,
                    layer.mlp.up_proj.lora_B.default.weight,
                    layer.mlp.down_proj.lora_A.default.weight,
                    layer.mlp.down_proj.lora_B.default.weight]
            else:
                params += [
                    *layer.cros_attn_qkv_proj.parameters(),
                    *layer.cros_attn_out_proj.parameters()]

        return params


    def forward(
            self, 
            input_ids, 
            labels=None,
            memory: torch.Tensor = None,
            prefill: bool = False
        ):
        assert input_ids.shape[-1] <= self.chunk_size
        outputs = self.decoder(
            input_ids=input_ids, 
            labels=labels,
            memory=memory,
            prefill=prefill)

        return outputs
