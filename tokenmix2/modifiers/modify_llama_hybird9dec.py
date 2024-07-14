import torch
import types
from .modify_llama import do_causal_flash_attn, do_hybird_attn, do_shift_mask_attn, do_prefill_accelerate_sdpa_attn
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast, repeat_kv, CrossEntropyLoss
from torch.utils.checkpoint import checkpoint

from peft import get_peft_model, LoraConfig, TaskType


def model_forward(
    self,
    input_ids: torch.LongTensor,
    labels: torch.Tensor = None,
    kv_caches: torch.Tensor = None,
    prefill: bool = False,
    generation: bool = False,
    sum_token: torch.Tensor = None,
    is_reduced: bool = False,
    **kwargs
):
    rets = self.model(
        input_ids=input_ids,
        kv_caches=kv_caches,
        prefill=prefill,
        generation=generation,
        sum_token=sum_token,
        is_reduced=is_reduced)
    
    if prefill:
        return rets
    else:
        hidden_states = rets

    if labels is not None:
        logits = checkpoint(self.lm_head, hidden_states, use_reentrant=False).float()
    else:
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
    prefill: bool = False,
    generation: bool = False,
    sum_token: torch.Tensor = None,
    is_reduced: bool = False,
):
    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    if kv_caches is None:
        kv_caches = [[None] * len(self.layers)] * 2

    if prefill:
        # NOTE: 新增加
        if self.retrieval == 'eos_token':
            prefix = sum_token.expand(hidden_states.shape[0], -1, -1)
            prefix = prefix.to(hidden_states.device)
            hidden_states = torch.cat([prefix, hidden_states], dim=-2)
        elif self.retrieval == 'avgpool':
            prefix = hidden_states.mean(dim=-2, keepdim=True)
            hidden_states = torch.cat([prefix, hidden_states], dim=-2)

        accum_keys = []
        accum_vals = []

    for decoder_layer, key_cache, val_cache in zip(self.layers, *kv_caches):
        if prefill:
            if torch.is_grad_enabled():
                keys, vals, hidden_states = checkpoint(
                    decoder_layer,
                    hidden_states,
                    (key_cache, val_cache),
                    prefill,
                    generation,
                    is_reduced,
                    use_reentrant=False)
            else:
                keys, vals, hidden_states = decoder_layer(
                    hidden_states,
                    (key_cache, val_cache),
                    prefill,
                    generation,
                    is_reduced)
            accum_keys.append(keys)
            accum_vals.append(vals)
        else:
            if torch.is_grad_enabled():
                hidden_states = checkpoint(
                    decoder_layer,
                    hidden_states,
                    (key_cache, val_cache),
                    prefill,
                    generation,
                    is_reduced,
                    use_reentrant=False)
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    (key_cache, val_cache),
                    prefill,
                    generation,
                    is_reduced)

    hidden_states = self.norm(hidden_states)

    if prefill:
        return (accum_keys, accum_vals)
    else:
        return hidden_states


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor = None,
    prefill: bool = False,
    generation: bool = False,
    is_reduced: bool = False,
):
    # self attention
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    if prefill:
        keys, vals, hidden_states = self.self_attn(hidden_states, kv_cache, prefill, generation, is_reduced)
    else:
        hidden_states = self.self_attn(hidden_states, kv_cache, prefill, generation, is_reduced)
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return (keys, vals, hidden_states) if prefill else hidden_states


def self_attn_forward(
    self,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    prefill: bool,
    generation: bool,
    is_reduced: bool,
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

    if not prefill:
        ques = self.q_proj(hidden_states).unflatten(-1, (num_heads,head_dim)).transpose(1,2)
        keys = self.k_proj(hidden_states).unflatten(-1, (num_kv_heads,head_dim)).transpose(1,2)
        vals = self.v_proj(hidden_states).unflatten(-1, (num_kv_heads,head_dim)).transpose(1,2)
        keys = repeat_kv(keys, num_kv_group)
        vals = repeat_kv(vals, num_kv_group)

        if generation:
            if hasattr(self, 'k_cache'):
                keys = torch.cat([self.k_cache, keys], dim=-2)
                vals = torch.cat([self.v_cache, vals], dim=-2)
            self.k_cache = keys.data
            self.v_cache = vals.data

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            if k_cache is not None and k_cache is not None:
                keys = torch.cat([k_cache, keys], dim=-2)
                vals = torch.cat([v_cache, vals], dim=-2)

        cos, sin = self.rotary_emb(vals, seq_len=128*1024)

        if generation:
            attn_output = do_causal_flash_attn(
                query=ques,
                key=keys,
                value=vals,
                cos=cos,
                sin=sin,
                out_proj=self.o_proj)
        else:
            if not is_reduced:
                attn_output = do_hybird_attn(
                    query=ques,
                    key=keys,
                    value=vals,
                    cos=cos,
                    sin=sin,
                    out_proj=self.o_proj,
                    chunk_size=self.chunk_size)
            else:
                attn_output = do_causal_flash_attn(
                    query=ques,
                    key=keys,
                    value=vals,
                    cos=cos,
                    sin=sin,
                    out_proj=self.o_proj)

        return attn_output

    else:
        if self.attn_mode == 'legacy':
            ques = self.q_proj(hidden_states[..., 1:, :]).unflatten(-1, (num_heads,head_dim)).transpose(1,2)
            keys = self.k_proj(hidden_states).unflatten(-1, (num_kv_heads,head_dim)).transpose(1,2)
            vals = self.v_proj(hidden_states).unflatten(-1, (num_kv_heads,head_dim)).transpose(1,2)
            keys = repeat_kv(keys, num_kv_group)
            vals = repeat_kv(vals, num_kv_group)
            cos, sin = self.rotary_emb(vals, seq_len=max_pos_embed)

            attn_output = do_causal_flash_attn(
                query=ques,
                key=keys,
                value=vals,
                cos=cos,
                sin=sin,
                out_proj=self.o_proj)
            
            if self.retrieval is not None:

                if self.retrieval == 'avgpool':
                    sum_token = hidden_states.mean(dim=-2, keepdim=True)
                    sum_token = sum_token.transpose(0,1)
                elif self.retrieval == 'eos_token':
                    sum_token = hidden_states[..., :1, :]
                    sum_token = sum_token.transpose(0,1)

                merge_ques = self.q_proj(sum_token).unflatten(-1, (num_heads, head_dim)).transpose(1,2)
                merge_keys = self.k_proj(sum_token).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)
                merge_vals = self.v_proj(sum_token).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)
                merge_keys = repeat_kv(merge_keys, num_kv_group)
                merge_vals = repeat_kv(merge_vals, num_kv_group)
                cos, sin = self.rotary_emb(merge_vals, seq_len=max_pos_embed)

                merge_outs = do_shift_mask_attn(
                    query=merge_ques,
                    key=merge_keys,
                    value=merge_vals,
                    cos=cos, sin=sin,
                    out_proj=self.o_proj,
                    shift_mask=True)

                merge_outs = merge_outs.transpose(0,1)
                attn_output = torch.cat([merge_outs, attn_output], dim=-2)

            keys = keys[..., 1:, :]
            vals = vals[..., 1:, :]

            return keys, vals, attn_output

        elif self.attn_mode == 'fast':
            assert self.retrieval is not None

            if self.retrieval == 'avgpool':
                sum_token = hidden_states.mean(dim=-2, keepdim=True)
                sum_token = sum_token.transpose(0,1)
            elif self.retrieval == 'eos_token':
                sum_token = hidden_states[..., :1, :]
                sum_token = sum_token.transpose(0,1)

            pad_length = hidden_states.shape[-2] - sum_token.shape[-2]
            pad_sequence = torch.zeros((1, pad_length, hidden_states.shape[-1]), dtype=sum_token.dtype, device=sum_token.device)
            sum_token = torch.cat([sum_token, pad_sequence], dim=-2)
            
            concat_states = torch.cat([hidden_states, sum_token], dim=0)
            concat_ques = self.q_proj(concat_states).unflatten(-1, (num_heads, head_dim)).transpose(1,2)
            concat_keys = self.k_proj(concat_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)
            concat_vals = self.v_proj(concat_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)
            concat_keys = repeat_kv(concat_keys, num_kv_group)
            concat_vals = repeat_kv(concat_vals, num_kv_group)

            keys = concat_keys[:-1, :, 1:, ...]
            vals = concat_vals[:-1, :, 1:, ...]
            cos, sin = self.rotary_emb(concat_vals, seq_len=max_pos_embed)

            outputs = do_prefill_accelerate_sdpa_attn(
                concat_ques, concat_keys, concat_vals,
                cos, sin, self.o_proj)
            
            ord_outs = outputs[:-1, 1:, :]
            mge_outs = outputs[-1:, ...].transpose(0,1)[:-pad_length, ...]
            outs = torch.cat([mge_outs, ord_outs], dim=-2)
            
            return keys, vals, outs
    
        else: raise NotImplementedError


class Decoder(torch.nn.Module):
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
            if hasattr(layer.self_attn, 'k_cache'):
                del layer.self_attn.k_cache
                del layer.self_attn.v_cache


    def __init__(
            self, 
            decoder, 
            chunk_size,
            enable_lora: bool = False,
            lora_kwargs: dict = None,
            retrieval: str = None,
            attn_mode: str = None):

        super().__init__()
        self.decoder = decoder
        self.chunk_size = chunk_size
        self.enable_lora = False
        self.retrieval = retrieval

        # 修改各种forward函数
        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)

        # NOTE: 新增加
        self.model.model.retrieval = retrieval

        for layer in self.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.chunk_size = chunk_size
            layer.self_attn.retrieval = retrieval
            layer.self_attn.attn_mode = attn_mode
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)

            # NOTE: 新增加
            if retrieval == 'eos_token':
                dtype = layer.self_attn.q_proj.weight.data.dtype
                device = layer.self_attn.q_proj.weight.data.device
                self.sum_token = torch.nn.Parameter(
                    torch.randn((1,1,4096), dtype=dtype, device=device),
                    requires_grad=True)
            else:
                self.sum_token = None


        self.enable_lora = enable_lora
        if self.enable_lora is True:
            self._init_lora(**lora_kwargs)


    def ft_params(self):
        params = []

        # NOTE: 新增加
        if self.retrieval == 'eos_token':
            params += [self.sum_token]

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

        return params


    def forward(
            self, 
            input_ids, 
            labels=None,
            kv_caches: torch.Tensor = None,
            prefill: bool = False,
            generation: bool = False,
            is_reduced: bool = False):

        outputs = self.decoder(
            input_ids=input_ids, 
            labels=labels, 
            kv_caches=kv_caches, 
            prefill=prefill,
            generation=generation,
            sum_token=self.sum_token,
            is_reduced=is_reduced,
            )

        return outputs
