from src.modifier import Modifier
from src.modifiers.modify_llama import do_causal_flash_attn
from peft import LoraConfig, get_peft_model, TaskType


import torch
from torch.utils.checkpoint import checkpoint
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast, repeat_kv

from profiler import WallTime


def model_forward(
    self,
    input_ids: torch.LongTensor,
    labels: torch.Tensor = None,
    kv_caches: torch.Tensor = None,
    **kwargs
):
    rets = self.model(
        input_ids=input_ids,
        kv_caches=kv_caches)

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
    kv_caches: torch.Tensor = None,
):
    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    if kv_caches is None:
        kv_caches = [[None] * len(self.layers)] * 2

    for decoder_layer, key_cache, val_cache in zip(self.layers, *kv_caches):
        hidden_states = decoder_layer(
            hidden_states,
            (key_cache, val_cache))

    hidden_states = self.norm(hidden_states)
    return hidden_states


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor = None
):
    # self attention
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states, kv_cache)
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    with WallTime.get('mlp'):     
        hidden_states = self.mlp(hidden_states)

    hidden_states = residual + hidden_states

    return hidden_states


def self_attn_forward(
    self,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
):
    num_heads, embed_dim = self.config.num_attention_heads, self.config.hidden_size
    head_dim = embed_dim // num_heads

    cond1 = hasattr(self.config, "max_sequence_length")
    cond2 = hasattr(self.config, "max_position_embeddings")
    cond3 = hasattr(self.config, "rope_scaling")
    max_pos_embed = int(max(
        self.config.max_sequence_length if cond1 else 0, 
        self.config.max_position_embeddings if cond2 else 0,
        self.config.max_position_embeddings * self.config.rope_scaling["factor"] if cond2 and cond3 else 0))

    num_kv_heads = self.config.num_key_value_heads
    num_kv_group = self.config.num_attention_heads // num_kv_heads

    with WallTime.get('qkv_proj'):
        ques = self.q_proj(hidden_states).unflatten(-1, (num_heads, head_dim)).transpose(1,2)
        keys = self.k_proj(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)
        vals = self.v_proj(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)

    if kv_cache is not None and kv_cache[0] is not None:
        k_cache, v_cache = kv_cache
        keys = torch.cat([k_cache, keys], dim=-2)
        vals = torch.cat([v_cache, vals], dim=-2)

    keys = repeat_kv(keys, num_kv_group)
    vals = repeat_kv(vals, num_kv_group)
    cos, sin = self.rotary_emb(vals, seq_len=128 * 1024)

    with WallTime.get('attn & oproj'):
        attn_output = do_causal_flash_attn(
            query=ques,
            key=keys,
            value=vals,
            cos=cos,
            sin=sin,
            out_proj=self.o_proj)
    
    return attn_output


class LlamaFlash(Modifier):
    def _init_lora(
            self,
            lora_rank: int, 
            lora_alpha: int, 
            lora_dropout: float):

        target_modules = r".*\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj"
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules)
        self.decoder = get_peft_model(self.model, peft_config)


    def __init__(self, model, save_ckp, load_ckp, config):
        super().__init__(model, save_ckp, load_ckp)
        self._init_lora(lora_rank=128, lora_alpha=512, lora_dropout=0)

        import types
        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)
        for layer in self.model.model.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)


    def ft_params(self):
        params = []

        for layer in self.model.base_model.layers:
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
            
        return params

    
    def reset(self):
        pass


    def forward(self, input_ids, labels=None, **kwargs):
        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0, 1)
        if labels is not None and labels.ndim == 3:
            labels = labels.flatten(0, 1)

        device = next(iter(self.model.parameters())).device
        input_ids = input_ids.to(device)

        return self.model(input_ids=input_ids, labels=labels)
