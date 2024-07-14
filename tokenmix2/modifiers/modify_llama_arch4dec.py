import torch
import types
from src.modifiers.modify_llama import do_causal_flash_attn, compute_loss
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
    memory_states = memory

    for decoder_layer in self.layers:
        hidden_states, memory_states = decoder_layer(
            hidden_states=hidden_states,
            memory_states=memory_states,
        )

    hidden_states = self.norm(hidden_states)

    return hidden_states


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    memory_states: torch.Tensor = None
):
    if memory_states is not None:
        memory_states = memory_states.to(hidden_states.device)
        hidden_states = torch.cat([memory_states, hidden_states], dim=-2)

    # self attention
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states)
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    if memory_states is not None:
        memory_tokens = memory_states.shape[-2]
        memory_states = hidden_states[:,:memory_tokens,:]
        hidden_states = hidden_states[:,memory_tokens:,:]

    return hidden_states, memory_states


def self_attn_forward(
    self,
    concat_states: torch.Tensor,
):
    ques = self.q_proj(concat_states).unflatten(-1, (32,128)).transpose(1,2)
    keys = self.k_proj(concat_states).unflatten(-1, (32,128)).transpose(1,2)
    vals = self.v_proj(concat_states).unflatten(-1, (32,128)).transpose(1,2)

    cos, sin = self.rotary_emb(vals, seq_len=4096)
    attn_output = do_causal_flash_attn(ques, keys, vals, cos, sin, self.o_proj)

    return attn_output


class Decoder(torch.nn.Module):
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
        

    def _create_adapter(self, layer, eos_token):
        kwargs = {
            "device": layer.self_attn.q_proj.weight.data.device,
            "dtype": layer.self_attn.q_proj.weight.data.dtype
        }
        layer.self_attn.adapter = torch.nn.Parameter(eos_token[None,None,:].expand(-1,4,-1).to(**kwargs), requires_grad=True)
        layer.self_attn.ada_gate = torch.nn.Parameter(torch.tensor(0, **kwargs), requires_grad=True)


    def __init__(self, decoder, chunk_size, eos_token):
        super().__init__()
        self.decoder = decoder
        self.chunk_size = chunk_size
        self.enable_lora = False

        # 修改各种forward函数
        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)
        for layer in self.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)
            self._create_adapter(layer, eos_token)


    def ft_params(self):
        params = []
        for layer in self.layers:
            params += [layer.self_attn.adapter, layer.self_attn.ada_gate]
        return params


    def forward(self, input_ids, memory=None, labels=None):
        assert input_ids.shape[-1] <= self.chunk_size
        outputs = self.decoder(input_ids=input_ids, memory=memory, labels=labels)
        return outputs