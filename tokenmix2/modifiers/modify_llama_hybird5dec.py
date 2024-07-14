import torch
import types
from src.modifiers.modify_llama import do_causal_flash_attn, do_hybird_attn
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
    sum_token: torch.Tensor = None
):
    inputs_embeds = checkpoint(self.embed_tokens, input_ids, use_reentrant=False)
    hidden_states = inputs_embeds

    if kv_caches is None:
        kv_caches = [None] * len(self.layers)

    if prefill:
        prefix_states = sum_token.expand(hidden_states.shape[0], -1, -1)
        posfix_states = sum_token.expand(hidden_states.shape[0], -1, -1)
        hidden_states = torch.cat([prefix_states, hidden_states, posfix_states], dim=-2)
        accum_keys = []
        accum_vals = []

    for decoder_layer, kv_cache in zip(self.layers, kv_caches):

        if prefill:
            keys, vals, hidden_states = checkpoint(
                decoder_layer,
                hidden_states,
                kv_cache,
                prefill,
                generation,
                use_reentrant=False)
            accum_keys.append(keys)
            accum_vals.append(vals)
        else:
            hidden_states = checkpoint(
                decoder_layer,
                hidden_states,
                kv_cache,
                prefill,
                generation,
                use_reentrant=False)

    hidden_states = checkpoint(self.norm, hidden_states, use_reentrant=False)

    if prefill:  
        accum_keys = [accum_key.to(accum_keys[0].device)[..., 1:-1, :] for accum_key in accum_keys]
        accum_vals = [accum_val.to(accum_vals[0].device)[..., 1:-1, :] for accum_val in accum_vals]

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

    ques = self.q_proj(hidden_states).unflatten(-1, (num_heads,head_dim)).transpose(1,2)
    keys = self.k_proj(hidden_states).unflatten(-1, (num_kv_heads,head_dim)).transpose(1,2)
    vals = self.v_proj(hidden_states).unflatten(-1, (num_kv_heads,head_dim)).transpose(1,2)
    keys = repeat_kv(keys, num_kv_group)
    vals = repeat_kv(vals, num_kv_group)

    if generation:
        # store & retrieval kv caches
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

    if prefill:
        """
        NOTE: write token inner chunk computation
        """
        attn_output = do_causal_flash_attn(
            query=ques,
            key=keys,
            value=vals,
            cos=cos,
            sin=sin,
            out_proj=self.o_proj)

        """
        NOTE: write token inter chunk computation
        """
        write_states = hidden_states[..., -1:, :].transpose(0, 1)
        write_ques = self.q_proj(write_states).unflatten(-1, (num_heads, head_dim)).transpose(1, 2)
        write_keys = self.k_proj(write_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1, 2)
        write_vals = self.v_proj(write_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1, 2)
        write_keys = repeat_kv(write_keys, num_kv_group)
        write_vals = repeat_kv(write_vals, num_kv_group)
        write_output = do_causal_flash_attn(
            query=write_ques,
            key=write_keys,
            value=write_vals,
            cos=cos,
            sin=sin,
            out_proj=self.o_proj)
        
        """
        NOTE: replace read token with write token
        """
        attn_output[..., :1, :] = write_output.transpose(0, 1)


        return keys, vals, attn_output

    elif generation:
        attn_output = do_causal_flash_attn(
            query=ques,
            key=keys,
            value=vals,
            cos=cos,
            sin=sin,
            out_proj=self.o_proj)
        
        attn_output
        
        return attn_output
    
    else:
        attn_output = do_hybird_attn(
            query=ques,
            key=keys,
            value=vals,
            cos=cos,
            sin=sin,
            out_proj=self.o_proj,
            chunk_size=self.chunk_size)
        
        attn_output
        
        return attn_output


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
            use_sdpa: bool = False):

        super().__init__()
        self.decoder = decoder
        self.chunk_size = chunk_size
        self.enable_lora = False

        # 修改各种forward函数
        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)
        for layer in self.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.chunk_size = chunk_size
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

        return params


    def forward(
            self, 
            input_ids, 
            labels=None,
            kv_caches: torch.Tensor = None,
            prefill: bool = False,
            generation: bool = False,
            sum_token: torch.Tensor = None):

        outputs = self.decoder(
            input_ids=input_ids, 
            labels=labels, 
            kv_caches=kv_caches, 
            prefill=prefill,
            generation=generation,
            sum_token=sum_token)

        return outputs
