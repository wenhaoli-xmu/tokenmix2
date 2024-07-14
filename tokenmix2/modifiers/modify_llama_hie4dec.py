import torch
import types
from src.modifiers.modify_llama import do_causal_flash_attn, compute_loss, ProjectHead
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast, LlamaRMSNorm

from peft import get_peft_model, TaskType, LoraConfig


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

    for layer_idx, decoder_layer in enumerate(self.layers):
        hidden_states = decoder_layer(
            hidden_states,
            memory_states=memory[layer_idx].unsqueeze(0) if memory is not None else None
        )

    hidden_states = self.norm(hidden_states)

    return hidden_states


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    memory_states: torch.Tensor = None
):
    # self attention
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states, memory_states)
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
    query_states = self.q_proj(hidden_states).unflatten(-1, (32,128)).transpose(1,2)
    key_states = self.k_proj(hidden_states).unflatten(-1, (32,128)).transpose(1,2)
    value_states = self.v_proj(hidden_states).unflatten(-1, (32,128)).transpose(1,2)

    if memory_states is not None:
        memory_states = memory_states.to(hidden_states.device)
        memory_keys, memory_vals = self.project_head(memory_states)
        key_states = torch.cat([memory_keys, key_states], dim=-2)
        value_states = torch.cat([memory_vals, value_states], dim=-2)

    cos, sin = self.rotary_emb(value_states, seq_len=4096)
    attn_output = do_causal_flash_attn(query_states, key_states, value_states, cos, sin, self.o_proj)

    return attn_output


class Decoder(torch.nn.Module):
    def _init_lora(self, lora_rank, lora_alpha, lora_dropout):
        decoder_peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=['key_proj', 'val_proj']
        )
        self.decoder = get_peft_model(self.decoder, decoder_peft_config)


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


    def __init__(self, decoder, chunk_size, enable_lora: bool, lora_kwargs: dict = None):
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
            layer.self_attn.project_head = ProjectHead(layer)

        # 是否使用lora微调
        self.enable_lora = enable_lora
        if self.enable_lora:
            self._init_lora(**lora_kwargs)


    def ft_params(self):
        params = []
        for layer in self.layers:
            if self.enable_lora:
                params += [
                    layer.self_attn.project_head.key_proj.lora_A.default.weight,
                    layer.self_attn.project_head.key_proj.lora_B.default.weight,
                    layer.self_attn.project_head.val_proj.lora_A.default.weight,
                    layer.self_attn.project_head.val_proj.lora_B.default.weight
                ]
            else:
                params += layer.self_attn.project_head.parameters()
        return params


    def forward(self, input_ids, memory=None, labels=None):
        assert input_ids.shape[-1] <= self.chunk_size
        outputs = self.decoder(input_ids=input_ids, memory=memory, labels=labels)
        return outputs