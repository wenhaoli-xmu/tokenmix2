import torch
import types
from src.modifiers.modify_llama import do_causal_flash_attn, do_hybird_attn, do_shift_mask_attn
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
    **kwargs
):
    if kv_caches is not None:
        assert kv_caches.ndim == 6 and kv_caches.shape[0] == 2
        kv_caches = kv_caches.transpose(0,1)
    
    rets = self.model(
        input_ids=input_ids,
        kv_caches=kv_caches,
        prefill=prefill,
        generation=generation,
        sum_token=sum_token)

    if prefill:
        return rets
    else:
        hidden_states = rets

    logits = checkpoint(self.lm_head, hidden_states, use_reentrant=False).float()

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
):
    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    if kv_caches is None:
        kv_caches = [None] * len(self.layers)

    if prefill:

        # ============================================================
        # NOTE: 新增加
        if self.retrieval == 'eos_token':
            postfix = sum_token.expand(hidden_states.shape[0], -1, -1)
            postfix = postfix.to(hidden_states.device)
            hidden_states = torch.cat([hidden_states, postfix], dim=-2)
        # ============================================================

        accum_keys = []
        accum_vals = []

    for decoder_layer, kv_cache in zip(self.layers, kv_caches):

        if prefill:
            keys, vals, hidden_states = decoder_layer(
                hidden_states,
                kv_cache,
                prefill,
                generation)
            accum_keys.append(keys)
            accum_vals.append(vals)
        else:
            hidden_states = decoder_layer(
                hidden_states,
                kv_cache,
                prefill,
                generation)

    hidden_states = self.norm(hidden_states)

    if prefill:

        # ==========================================================================================
        # NOTE: 新增加
        if self.retrieval == 'eos_token' and self.distribute == 'prefix':
            accum_keys = [accum_key.to(accum_keys[0].device)[...,:-1,:] for accum_key in accum_keys]
            accum_vals = [accum_val.to(accum_vals[0].device)[...,:-1,:] for accum_val in accum_vals]
        else:
            accum_keys = [accum_key.to(accum_keys[0].device) for accum_key in accum_keys]
            accum_vals = [accum_val.to(accum_vals[0].device) for accum_val in accum_vals]
        # ==========================================================================================


        keys = torch.stack(accum_keys, dim=0)
        vals = torch.stack(accum_vals, dim=0)
        rets = torch.stack((keys, vals), dim=0)

        assert rets.ndim == 6 and rets.shape[0] == 2
        return rets
    else:
        return hidden_states


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor = None,
    prefill: bool = False,
    generation: bool = False
):
    # self attention
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    if prefill:
        keys, vals, hidden_states = self.self_attn(hidden_states, kv_cache, prefill, generation)
    else:
        hidden_states = self.self_attn(hidden_states, kv_cache, prefill, generation)
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
):
    num_heads, embed_dim = self.config.num_attention_heads, self.config.hidden_size
    head_dim = embed_dim // num_heads
    max_pos_embed = self.config.max_position_embeddings
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
            keys = torch.cat([k_cache, keys], dim=-2)
            vals = torch.cat([v_cache, vals], dim=-2)

        cos, sin = self.rotary_emb(vals, seq_len=max_pos_embed)

        if generation:
            attn_output = do_causal_flash_attn(
                query=ques,
                key=keys,
                value=vals,
                cos=cos,
                sin=sin,
                out_proj=self.o_proj)
        else:
            attn_output = do_hybird_attn(
                query=ques,
                key=keys,
                value=vals,
                cos=cos,
                sin=sin,
                out_proj=self.o_proj,
                chunk_size=self.chunk_size)

        return attn_output

    else:

        # =========================================================================================================
        # NOTE: 新增加
        if self.retrieval is not None:

            if self.retrieval == 'eos_token':
                r_ques = self.q_proj(hidden_states[..., -1:, :]).unflatten(-1, (num_heads, head_dim)).transpose(1,2)
                r_keys = self.k_proj(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)
                r_vals = self.v_proj(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)
                r_keys = repeat_kv(r_keys, num_kv_group)
                r_vals = repeat_kv(r_vals, num_kv_group)
                cos, sin = self.rotary_emb(r_vals, seq_len=max_pos_embed)
                r_outs = do_causal_flash_attn(
                    query=r_ques,
                    key=r_keys,
                    value=r_vals,
                    cos=cos, sin=sin,
                    out_proj=self.o_proj)

                """此时的sum token中已经融合了本chunk内的信息"""
                sum_token = r_outs.transpose(0,1)

            elif self.retrieval == 'avgpool':
                sum_token = hidden_states.mean(dim=-2, keepdim=True)
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

            if self.distribute == 'prefix':
                hidden_states = torch.cat([merge_outs, hidden_states], dim=-2)
        # =========================================================================================================

        
        ques = self.q_proj(hidden_states).unflatten(-1, (num_heads,head_dim)).transpose(1,2)
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


        # ============================================
        # NOTE: 新增加
        if self.retrieval is not None:
            if self.distribute == 'prefix':
                attn_output = attn_output[..., 1:, :]
                keys = keys[..., 1:, :]
                vals = vals[..., 1:, :]

            elif self.distribute == 'add':
                attn_output += merge_outs
        # ============================================
  

        return keys, vals, attn_output


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
            distribute: str = None):

        super().__init__()
        self.decoder = decoder
        self.chunk_size = chunk_size
        self.enable_lora = False
        self.retrieval = retrieval
        self.distribute = distribute

        # 修改各种forward函数
        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)
        
        # ======================================
        # NOTE: 新增加
        self.model.model.retrieval = retrieval
        self.model.model.distribute = distribute
        # ======================================

        for layer in self.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.chunk_size = chunk_size
            layer.self_attn.retrieval = retrieval
            layer.self_attn.distribute = distribute
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)

            # ==========================================================
            # NOTE: 新增加
            if retrieval == 'eos_token':
                dtype = layer.self_attn.q_proj.weight.data.dtype
                device = layer.self_attn.q_proj.weight.data.device
                self.sum_token = torch.nn.Parameter(
                    torch.randn((1,1,4096), dtype=dtype, device=device),
                    requires_grad=True)
            else:
                self.sum_token = None
            # ==========================================================


        self.enable_lora = enable_lora
        if self.enable_lora is True:
            self._init_lora(**lora_kwargs)


    def ft_params(self):
        params = []

        # ===============================
        # NOTE: 新增加
        if self.retrieval == 'eos_token':
            params += [self.sum_token]
        # ===============================

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
            generation: bool = False):

        outputs = self.decoder(
            input_ids=input_ids, 
            labels=labels, 
            kv_caches=kv_caches, 
            prefill=prefill,
            generation=generation,
            sum_token=self.sum_token  # NOTE: 新增加
            )

        return outputs
