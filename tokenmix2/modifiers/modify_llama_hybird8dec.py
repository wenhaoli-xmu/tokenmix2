import torch
import types
from src.modifiers.modify_llama import do_causal_flash_attn, do_hybird_attn, do_shift_mask_attn, do_highlv_attn, QKVProj, OProj
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
            highlv_states = sum_token.expand(hidden_states.shape[0], -1, -1)
            highlv_states = highlv_states.to(hidden_states.device)
        elif self.retrieval == 'avgpool':
            highlv_states = hidden_states.mean(dim=-2, keepdim=True)
        else:
            raise NotImplementedError
        # ============================================================

        accum_keys = []
        accum_vals = []
    else:
        highlv_states = None

    for decoder_layer, kv_cache in zip(self.layers, kv_caches):

        if prefill:
            keys, vals, hidden_states, highlv_states = decoder_layer(
                hidden_states,
                highlv_states,
                kv_cache,
                prefill,
                generation)
            accum_keys.append(keys)
            accum_vals.append(vals)
        else:
            hidden_states, highlv_states = decoder_layer(
                hidden_states,
                highlv_states,
                kv_cache,
                prefill,
                generation)

    hidden_states = self.norm(hidden_states)

    if prefill:
        accum_keys = [key.to(keys[0].device) for key in accum_keys]
        accum_vals = [val.to(vals[0].device) for val in accum_vals]
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
    highlv_states: torch.Tensor,
    kv_cache: torch.Tensor = None,
    prefill: bool = False,
    generation: bool = False
):
    num_hidden = hidden_states.shape[-2]

    def concat(hidden_states, highlv_states):
        return torch.cat([hidden_states, highlv_states], dim=-2) if highlv_states is not None else hidden_states

    def deconcat(concat_states):
        nonlocal num_hidden
        if concat_states.shape[-2] == num_hidden:
            return concat_states, None
        else:
            return concat_states[..., :num_hidden, :], concat_states[..., num_hidden:, :]

    concat_states = concat(hidden_states, highlv_states)
    residual = concat_states
    concat_states = self.input_layernorm(concat_states)

    if prefill:
        keys, vals, concat_states = self.self_attn(*deconcat(concat_states), kv_cache, prefill, generation)
    else:
        concat_states = self.self_attn(*deconcat(concat_states), kv_cache, prefill, generation)

    concat_states = concat(hidden_states, highlv_states)
    concat_states = residual + concat_states

    # Fully Connected
    residual = concat_states
    concat_states = self.post_attention_layernorm(concat_states)
    concat_states = self.mlp(concat_states)
    concat_states = residual + concat_states

    return (keys, vals, *deconcat(concat_states)) if prefill else (*deconcat(concat_states),)


def self_attn_forward(
    self,
    hidden_states: torch.Tensor,
    highlv_states: torch.Tensor,
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

        """
        首先是将所有的highlv_states进行分发
        """
        inforich_states = torch.cat([highlv_states, hidden_states], dim=-2)
        inforich_ques = self.q_proj(inforich_states[..., 1:, :]).unflatten(-1, (num_heads, head_dim)).transpose(1,2)
        inforich_keys = self.k_proj(inforich_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)
        inforich_vals = self.v_proj(inforich_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)
        inforich_keys = repeat_kv(inforich_keys, num_kv_group)
        inforich_vals = repeat_kv(inforich_vals, num_kv_group)

        cos, sin = self.rotary_emb(inforich_vals, seq_len=max_pos_embed)
        inforich_outs = do_causal_flash_attn(
            query=inforich_ques,
            key=inforich_keys,
            value=inforich_vals,
            cos=cos, sin=sin,
            out_proj=self.o_proj)
        
        """
        然后是对这些highlv_states进行单独计算
        """
        assert highlv_states.shape[1] == 1
        highlv_states = highlv_states.transpose(0,1)
        highlv_ques, highlv_keys, highlv_vals = self.qkv_proj(highlv_states, num_heads, num_kv_heads, head_dim)
        highlv_keys = repeat_kv(highlv_keys, num_kv_group)
        highlv_vals = repeat_kv(highlv_vals, num_kv_group)

        if self.history_expand == "none":
            pass
        elif self.history_expand in ("stepwise", "segment"):
            expand_states = hidden_states.flatten(0,1).unsqueeze(0)
            expand_keys = self.k_proj(expand_states).unflatten(-1, (num_heads, head_dim)).transpose(1,2)
            expand_vals = self.v_proj(expand_states).unflatten(-1, (num_heads, head_dim)).transpose(1,2)
            highlv_keys = torch.cat([expand_keys, highlv_keys], dim=-2)
            highlv_vals = torch.cat([expand_vals, highlv_vals], dim=-2)
        else:
            raise NotImplementedError

        highlv_outs = do_highlv_attn(
            query=highlv_ques,
            key=highlv_keys,
            value=highlv_vals,
            cos=cos, sin=sin,
            out_proj=None,
            expand_type=self.history_expand)
    
        highlv_outs = self.out_proj(highlv_outs)
        highlv_outs = highlv_outs.transpose(0,1)

        attn_outs = torch.cat([inforich_outs, highlv_outs], dim=-2)

        if prefill:
            keys = inforich_keys[..., 1:, :]
            vals = inforich_vals[..., 1:, :]
            return keys, vals, attn_outs
        else:
            return attn_outs


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
            history_expand: str = None):

        super().__init__()
        self.decoder = decoder
        self.chunk_size = chunk_size
        self.enable_lora = False
        self.retrieval = retrieval
        self.history_expand = history_expand

        # 修改各种forward函数
        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)
        
        # ======================================
        # NOTE: 新增加
        self.model.model.retrieval = retrieval
        # ======================================

        for layer in self.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.chunk_size = chunk_size
            layer.self_attn.retrieval = retrieval
            layer.self_attn.history_expand = history_expand
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)
            layer.self_attn.qkv_proj = QKVProj(layer)
            layer.self_attn.out_proj = OProj(layer, zero_init=False)

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
                    *layer.self_attn.qkv_proj.get_lora_parameters(),
                    *layer.self_attn.out_proj.get_lora_parameters()]

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
