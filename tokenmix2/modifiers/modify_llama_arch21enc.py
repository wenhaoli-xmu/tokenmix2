import torch
from torch import nn
import types
from peft import get_peft_model, LoraConfig, TaskType

from transformers import T5EncoderModel
from transformers.models.t5.modeling_t5 import (
    T5Model,
    T5LayerFF,
    T5LayerSelfAttention)


def model_forward(
    self,
    inputs_embeds: torch.Tensor,
    memory: torch.Tensor = None,
    **kwargs
):
    new_memory = self.encoder(
        inputs_embeds=inputs_embeds,
        memory=memory)

    if memory is not None:
        memory = memory.to(new_memory.device)
        new_memory = new_memory + memory

    return new_memory.cpu()


def stack_forward(
    self,
    inputs_embeds: torch.Tensor,
    memory: torch.Tensor = None
):
    position_bias = None

    hidden_states = inputs_embeds
    if memory is not None:
        hidden_states = torch.cat([memory, hidden_states], dim=-2)
    
    hidden_states = self.dropout(hidden_states)

    for layer_module in self.block:
        hidden_states, position_bias = layer_module(
            hidden_states,
            position_bias=position_bias)

    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    if memory is not None:
        memory, hidden_states = hidden_states.chunk(2, dim=-2)

    return hidden_states


def block_forward(
    self,
    hidden_states,
    position_bias=None,
):
    # self attention
    hidden_states, position_bias = self.layer[0](
        hidden_states,
        position_bias=position_bias)

    # feed forward network
    hidden_states = self.layer[-1](hidden_states)

    return hidden_states, position_bias


def self_attn_forward(
    self,
    hidden_states,
    position_bias=None
):
    normed_hidden_states = self.layer_norm(hidden_states)
    attn_output, position_bias = self.SelfAttention(
        normed_hidden_states,
        position_bias=position_bias,
    )
    hidden_states = hidden_states + self.dropout(attn_output)
    return hidden_states, position_bias


def attn_forward(
    self,
    hidden_states,
    key_value_states=None,
    position_bias=None
):
    batch_size, seq_length = hidden_states.shape[:2]

    real_seq_length = seq_length
    key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

    def shape(states):
        """projection"""
        return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def unshape(states):
        """reshape"""
        return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    def project(hidden_states, proj_layer, key_value_states):
        if key_value_states is None:
            hidden_states = shape(proj_layer(hidden_states))
        else:
            hidden_states = shape(proj_layer(key_value_states))
        return hidden_states

    query_states = shape(self.q(hidden_states))
    key_states = project(hidden_states, self.k, key_value_states)
    value_states = project(hidden_states, self.v, key_value_states)

    scores = torch.matmul(query_states, key_states.transpose(3, 2))

    if position_bias is None:
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

    position_bias_masked = position_bias
    scores += position_bias_masked
    attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        scores
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.dropout, training=self.training
    )

    attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    attn_output = self.o(attn_output)

    return attn_output, position_bias


class Encoder(torch.nn.Module):
    def _init_lora(
            self, 
            lora_rank: int, 
            lora_alpha: int, 
            lora_dropout: float
        ):
        target_modules = r".*\.EncDecAttention\.(q|k|v|o)"
        encoder_peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules
        )
        self.encoder = get_peft_model(self.encoder, encoder_peft_config)


    @property
    def blocks(self):
        if self.enable_lora:
            return self.encoder.base_model.model.encoder.block
        else:
            return self.encoder.encoder.block


    @property
    def model(self):
        if self.enable_lora:
            return self.encoder.base_model.model
        else:
            return self.encoder


    def __init__(
            self, 
            chunk_size,
            enable_lora: bool, 
            lora_kwargs: dict = None,
        ):

        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(
            "google/flan-t5-large", 
            torch_dtype=torch.bfloat16)
        decoder = T5Model.from_pretrained(
            "google/flan-t5-large", 
            torch_dtype=torch.bfloat16)

        self.chunk_size = chunk_size
        self.enable_lora = False

        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.encoder.forward = types.MethodType(stack_forward, self.model.encoder)

        for block_idx, block in enumerate(self.blocks):
            assert isinstance(block.layer[0], T5LayerSelfAttention)
            assert isinstance(block.layer[1], T5LayerFF)
            block.forward = types.MethodType(block_forward, block)
            block.layer[0].forward = types.MethodType(self_attn_forward, block.layer[0])
            block.layer[0].SelfAttention.forward = types.MethodType(attn_forward, block.layer[0].SelfAttention)

            if block_idx == 0:
                assert block.layer[0].SelfAttention.has_relative_attention_bias
            else:
                assert not block.layer[0].SelfAttention.has_relative_attention_bias

        del decoder

        self.enable_lora = enable_lora
        if self.enable_lora:
            self._init_lora(**lora_kwargs)


    def ft_params(self):
        params = list(self.blocks[0].layer[0].SelfAttention.relative_attention_bias.parameters())

        for block in self.blocks:
            if self.enable_lora:
                params += [
                    block.layer[0].SelfAttention.q.lora_A.default.weight,
                    block.layer[0].SelfAttention.q.lora_B.default.weight,
                    block.layer[0].SelfAttention.v.lora_A.default.weight,
                    block.layer[0].SelfAttention.v.lora_B.default.weight]
            else:
                params += block.layer[0].SelfAttention.parameters()
            
        return params


    def forward(
            self,
            input_ids: torch.Tensor, 
            memory: torch.Tensor
        ):
        inputs_embeds = self.model.encoder.embed_tokens(input_ids).cpu()
        memory = self.encoder(inputs_embeds=inputs_embeds, memory=memory).cpu()
        return memory
    
