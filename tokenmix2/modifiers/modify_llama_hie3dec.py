import torch
import types
from src.modifiers.modify_llama import do_causal_flash_attn, compute_loss
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast, LlamaRMSNorm, CrossEntropyLoss


def model_forward(
    self,
    input_ids: torch.LongTensor,
    labels: torch.Tensor = None,
    memory: torch.Tensor = None
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

    for layer_idx, decoder_layer in enumerate(self.layers):
        hidden_states = decoder_layer(
            hidden_states,
            memory_layer=memory[layer_idx].unsqueeze(0) if memory is not None else None
        )

    hidden_states = self.norm(hidden_states)

    return hidden_states


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    memory_layer: torch.Tensor = None
):
    """
    self attention
    """
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(
        hidden_states=hidden_states,
    )
    hidden_states = residual + hidden_states

    """
    cross attention
    """
    if memory_layer is not None:
        residual = hidden_states
        hidden_states = self.input_layernorm2(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            memory_states=memory_layer
        )
        hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


def self_attn_forward(
    self,
    hidden_states: torch.Tensor,
    memory_states: torch.Tensor = None
):
    if memory_states is None:
        # NOTE: self attention
        query_states = self.q_proj(hidden_states).unflatten(-1, (32,128)).transpose(1,2)
        key_states = self.k_proj(hidden_states).unflatten(-1, (32,128)).transpose(1,2)
        value_states = self.v_proj(hidden_states).unflatten(-1, (32,128)).transpose(1,2)

        cos, sin = self.rotary_emb(value_states, seq_len=4096)
        attn_output = do_causal_flash_attn(query_states, key_states, value_states, cos, sin, self.o_proj)
    else:
        # NOTE: cross attention
        memory_states = memory_states.to(hidden_states.device)
        query_states = self.q_proj2(hidden_states).unflatten(-1, (32,128)).transpose(1,2)
        key_states = self.k_proj2(memory_states).unflatten(-1, (32,128)).transpose(1,2)
        value_states = self.v_proj2(memory_states).unflatten(-1, (32,128)).transpose(1,2)

        cos, sin = self.rotary_emb(value_states, seq_len=4096)
        attn_output = do_causal_flash_attn(query_states, key_states, value_states, cos, sin, self.o_proj2)

    return attn_output


class Decoder(torch.nn.Module):
    def __init__(self, decoder, chunk_size, wo_zero_init):
        super().__init__()
        self.decoder = decoder
        self.chunk_size = chunk_size

        decoder.forward = types.MethodType(model_forward, decoder)
        decoder.model.forward = types.MethodType(model_model_forward, decoder.model)
        for layer in decoder.model.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)

            kwargs = {
                "device": layer.self_attn.q_proj.weight.data.device,
                "dtype": layer.self_attn.q_proj.weight.data.dtype
            }
            layer.self_attn.q_proj2 = torch.nn.Linear(4096,4096,bias=False,**kwargs)
            layer.self_attn.k_proj2 = torch.nn.Linear(4096,4096,bias=False,**kwargs)
            layer.self_attn.v_proj2 = torch.nn.Linear(4096,4096,bias=False,**kwargs)
            layer.self_attn.o_proj2 = torch.nn.Linear(4096,4096,bias=False,**kwargs)
            layer.input_layernorm2 = LlamaRMSNorm(4096, 1e-5)

            layer.self_attn.q_proj2.weight.data = layer.self_attn.q_proj.weight.data.clone()
            layer.self_attn.k_proj2.weight.data = layer.self_attn.k_proj.weight.data.clone()
            layer.self_attn.v_proj2.weight.data = layer.self_attn.v_proj.weight.data.clone()
            layer.self_attn.o_proj2.weight.data = layer.self_attn.o_proj.weight.data.clone()

            if wo_zero_init:
                layer.self_attn.o_proj2.weight.data.fill_(0)

            layer.input_layernorm2.weight.data = layer.input_layernorm.weight.data.clone()


    def ft_params(self):
        params = []
        for layer in self.decoder.model.layers:
            params += [
                layer.self_attn.q_proj2.weight,
                layer.self_attn.k_proj2.weight,
                layer.self_attn.v_proj2.weight,
                layer.self_attn.o_proj2.weight,
                *layer.input_layernorm2.parameters()
            ]
        return params


    def forward(self, input_ids, memory=None, labels=None):
        outputs = self.decoder(input_ids=input_ids, memory=memory, labels=labels)
        return outputs