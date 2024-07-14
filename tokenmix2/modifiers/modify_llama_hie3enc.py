import torch
import types
from src.modifiers.modify_llama import do_causal_flash_attn


def model_forward(
    self,
    inputs_embeds: torch.Tensor,
    memory: torch.Tensor,
    merge_tokens: torch.Tensor
):
    memory = self.model(
        inputs_embeds=inputs_embeds,
        memory=memory,
        merge_tokens=merge_tokens
    )

    return memory


def model_model_forward(
    self,
    inputs_embeds: torch.Tensor,
    memory: torch.Tensor,
    merge_tokens: torch.Tensor
):
    assert memory.ndim == 3 and memory.shape[0] == 32 and memory.shape[-1] == 4096

    hidden_states = inputs_embeds
    updated_memory = []

    for decoder_layer, memory_states in zip(self.layers, memory.chunk(32)):
        updated_memory += [merge_tokens.cpu()]

        hidden_states, merge_tokens = decoder_layer(
            hidden_states=hidden_states,
            memory_states=memory_states,
            merge_tokens=merge_tokens
        )

    return torch.cat(updated_memory, dim=0)


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    memory_states: torch.Tensor,
    merge_tokens: torch.Tensor
):
    memory_states = memory_states.to(hidden_states.device)
    merge_tokens = memory_states.to(hidden_states.device)
    full_states = torch.cat([memory_states, hidden_states, merge_tokens], dim=-2)

    residual = full_states
    full_states = self.input_layernorm(full_states)
    full_states = self.self_attn(full_states)
    full_states = residual + full_states

    residual = full_states
    full_states = self.post_attention_layernorm(full_states)
    full_states = self.mlp(full_states)
    full_states = residual + full_states

    memory_states, hidden_states, merge_tokens = full_states.chunk(3, dim=-2)

    return hidden_states, merge_tokens


def attn_forward(
    self,
    full_states: torch.Tensor
):
    query_states = self.q_proj(full_states).unflatten(-1, (32,128)).transpose(1,2)
    key_states = self.k_proj(full_states).unflatten(-1, (32,128)).transpose(1,2)
    value_states = self.v_proj(full_states).unflatten(-1, (32,128)).transpose(1,2)

    cos, sin = self.rotary_emb(value_states, seq_len=4096)
    attn_output = do_causal_flash_attn(query_states, key_states, value_states, cos, sin, self.o_proj)

    return attn_output


class Encoder(torch.nn.Module):
    def __init__(self, encoder, chunk_size):
        super().__init__()
        self.encoder = encoder
        self.chunk_size = chunk_size

        encoder.forward = types.MethodType(model_forward, encoder)
        encoder.model.forward = types.MethodType(model_model_forward, encoder.model)
        for layer in encoder.model.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(attn_forward, layer.self_attn)

    def ft_params(self):
        """
        可调参数目前只有Wq,Wk,Wv,Wo
        """
        params = []
        for layer in self.encoder.model.layers:
            params += [
                layer.self_attn.q_proj.weight,
                layer.self_attn.k_proj.weight,
                layer.self_attn.v_proj.weight,
                layer.self_attn.o_proj.weight
            ]
        return params
    
    def forward(self, input_ids, memory, merge_tokens):
        """
        memory <- encoder(input_ids, memory)  # update the memory
        """
        inputs_embeds = self.encoder.model.embed_tokens(input_ids).cpu()
        memory = self.encoder(inputs_embeds=inputs_embeds, memory=memory, merge_tokens=merge_tokens)
        return memory