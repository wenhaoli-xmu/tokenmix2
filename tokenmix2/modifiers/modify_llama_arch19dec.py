import torch
import types
from src.modifiers.modify_llama import compute_loss, do_causal_flash_attn, CrossAttnQKVProj, OProj, LlamaRMSNorm 
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast


def model_forward(
    self,
    input_ids: torch.LongTensor,
    labels: torch.Tensor = None,
    memory: torch.Tensor = None,
    **kwargs
):
    hidden_states = self.model(
        input_ids=input_ids,
        memory=memory
    )
    logits = self.lm_head(hidden_states).float()

    if labels is not None:
        loss, _, valid_token_num = compute_loss(logits, labels, shift=False)
        print(f"my loss: {loss.item()}", flush=True)
        loss = loss * valid_token_num
    else:
        loss = None

    return CausalLMOutputWithPast(loss=loss, logits=logits)


def model_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    memory: torch.Tensor = None
):
    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    for decoder_layer in self.layers:
        hidden_states = decoder_layer(
            hidden_states=hidden_states,
            memory=memory,
        )

    hidden_states = self.norm(hidden_states)

    return hidden_states


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
        residual = hidden_states
        hidden_states = self.cros_attn_norm(hidden_states)
        hidden_states = cross_attn(
            qkv_proj=self.cros_attn_qkv_proj, 
            out_proj=self.cros_attn_out_proj,
            rotary_emb=self.self_attn.rotary_emb,
            hidden_states=hidden_states, 
            memory=memory)
        hidden_states = residual + hidden_states    

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


def cross_attn(
        qkv_proj: CrossAttnQKVProj,
        out_proj: OProj,
        rotary_emb: torch.nn.Module,
        hidden_states: torch.Tensor,
        memory: torch.Tensor
) -> torch.FloatTensor:
    ques, keys, vals = qkv_proj(hidden_states, memory)
    cos, sin = rotary_emb(vals, seq_len=4096)
    return do_causal_flash_attn(ques, keys, vals, cos, sin, out_proj)


def self_attn_forward(
    self,
    hidden_states: torch.Tensor
):
    ques = self.q_proj(hidden_states).unflatten(-1, (32,128)).transpose(1,2)
    keys = self.k_proj(hidden_states).unflatten(-1, (32,128)).transpose(1,2)
    vals = self.v_proj(hidden_states).unflatten(-1, (32,128)).transpose(1,2)

    if hasattr(self, 'k_cache'):
        keys = torch.cat([self.k_cache, keys], dim=-2)
        vals = torch.cat([self.v_cache, vals], dim=-2)
    self.k_cache = keys.data
    self.v_cache = vals.data

    cos, sin = self.rotary_emb(vals, seq_len=4096)
    attn_output = do_causal_flash_attn(
        query=ques,
        key=keys,
        value=vals,
        cos=cos,
        sin=sin,
        out_proj=self.o_proj
    )

    return attn_output


class Decoder(torch.nn.Module):
    @property
    def layers(self):
        return self.decoder.model.layers


    @property
    def model(self):
        return self.decoder


    def reset(self):
        for layer in self.layers:
            if hasattr(layer.self_attn, 'k_cache'):
                del layer.self_attn.k_cache
                del layer.self_attn.v_cache

    
    def clear_kv_cache(self, num_kv_cache):
        for layer in self.layers:
            layer.self_attn.k_cache = layer.self_attn.k_cache[...,:-num_kv_cache,:]
            layer.self_attn.v_cache = layer.self_attn.v_cache[...,:-num_kv_cache,:]


    def __init__(
            self, 
            decoder, 
            chunk_size,
        ):
        super().__init__()
        self.decoder = decoder
        self.chunk_size = chunk_size

        # 修改各种forward函数
        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)
        for layer in self.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)
            layer.cros_attn_qkv_proj = CrossAttnQKVProj(layer)
            layer.cros_attn_out_proj = OProj(layer, zero_init=True)
            layer.cros_attn_norm = LlamaRMSNorm(layer, 4096, 1e-6)


    def ft_params(self):
        params = []
        for layer in self.layers:
            params += layer.cros_attn_qkv_proj.parameters()
            params += layer.cros_attn_out_proj.parameters()
        return params


    def forward(
            self, 
            input_ids, 
            memory=None,
            labels=None
        ):
        assert input_ids.shape[-1] <= self.chunk_size
        outputs = self.decoder(input_ids=input_ids, memory=memory, labels=labels)
        return outputs
