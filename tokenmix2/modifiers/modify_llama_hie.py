import math, types, warnings
from typing import Optional, Tuple, Union, List

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.models.llama.modeling_llama import (
    rotate_half,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    DynamicCache,
    _prepare_4d_causal_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask,
    CrossEntropyLoss,
    repeat_kv,
    LlamaRMSNorm
    )
from transformers.cache_utils import Cache

from transformers.cache_utils import Cache
import torch.nn.functional as F

from src.modifier import Modifier, SegmentRecurrentModifier
from functools import partial

from flash_attn import flash_attn_func
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from copy import deepcopy
import builtins


def fake_print(*args, **kwargs):
    pass


def apply_rotary_pos_emb(mat, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    mat_embed = (mat * cos) + (rotate_half(mat) * sin)

    return mat_embed


def new_posid(num_token: int, device, dtype, bsz):
    appendix = torch.arange(num_token, device=device)
    appendix = appendix[None,:].expand(bsz, -1)
    return appendix


def check_and_apply_rope(query, key, value, cos, sin):
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2]

    assert key.shape == (batch_size, num_heads, num_kv, head_dim)
    assert value.shape == (batch_size, num_heads, num_kv, head_dim)

    new_posid_spec = partial(new_posid, device=query.device, dtype=query.dtype, bsz=batch_size)

    Q = apply_rotary_pos_emb(query, cos, sin, new_posid_spec(num_kv)[:,-num_query:])
    K = apply_rotary_pos_emb(key, cos, sin, new_posid_spec(num_kv))
    V = value

    return Q, K, V


def check_and_apply_offset_rope(query, key, value, cos, sin):
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2]

    assert key.shape == (batch_size, num_heads, num_kv, head_dim)
    assert value.shape == (batch_size, num_heads, num_kv, head_dim)

    new_posid_spec = partial(new_posid, device=query.device, dtype=query.dtype, bsz=batch_size)

    Q = apply_rotary_pos_emb(query, cos, sin, new_posid_spec(num_kv + num_query)[:,-num_query:])
    K = apply_rotary_pos_emb(key, cos, sin, new_posid_spec(num_kv))
    V = value

    # 这里需要检查一下生成的position embedding到底一不一样

    return Q, K, V


def do_causal_flash_attn(query, key, value, cos, sin, out_proj: torch.nn.Linear = None):
    """
    仅仅支持下三角形的attention mask
    """
    batch_size, num_heads, num_query, head_dim = query.shape
    Q, K, V = check_and_apply_rope(query, key, value, cos, sin)
    Q, K, V = Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2)

    attn_output = flash_attn_func(
        Q, K, V, causal=True
    )
    attn_output = attn_output.reshape(batch_size, num_query, num_heads * head_dim).contiguous()
    
    if out_proj is not None:
        attn_output = out_proj(attn_output)

    return attn_output


def do_offset_flash_attn(query, key, value, cos, sin, out_proj: torch.nn.Linear = None):
    """
    仅仅支持下三角形的attention mask
    """
    batch_size, num_heads, num_query, head_dim = query.shape
    Q, K, V = check_and_apply_offset_rope(query, key, value, cos, sin)
    Q, K, V = Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2)

    attn_output = flash_attn_func(
        Q, K, V, causal=False
    )
    attn_output = attn_output.reshape(batch_size, num_query, num_heads * head_dim).contiguous()
    
    if out_proj is not None:
        attn_output = out_proj(attn_output)

    return attn_output


def decoder_attn_forward(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        trigger: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )     

        bsz, q_len = hidden_states.shape[:2]

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            if trigger:
                # trigger代表该attn是用来处理history的，他们的rope添加方式有所区别
                assert memory_states is not None
                query_states = self.q_proj2(hidden_states)
                key_states = self.k_proj2(memory_states)
                value_states = self.v_proj2(memory_states)
            else:
                assert memory_states is None
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

            if use_cache:
                assert trigger is False
                assert hidden_states.shape[-2] <= self.chunk_size

                # NOTE: 这一部分用来缓存hidden states
                if hasattr(self, "state_cache"):
                    self.state_cache = torch.cat([self.state_cache, hidden_states], dim=-2)
                else:
                    self.state_cache = hidden_states

                # # NOTE: 这一部分是在causal generation的时候起作用的
                # if hasattr(self, "k_cache"):
                #     key_states = torch.cat([self.k_cache, key_states.data], dim=-2)
                #     value_states = torch.cat([self.v_cache, value_states.data], dim=-2)
                # self.k_cache = key_states.data
                # self.v_cache = value_states.data
                    

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # modified: 将这里的seq_len=kv_seq_len修改为2048
        cos, sin = self.rotary_emb(value_states, seq_len=4096)

        if trigger:
            attn_output = do_offset_flash_attn(query_states, key_states, value_states, cos, sin, self.o_proj2)
        else:
            attn_output = do_causal_flash_attn(query_states, key_states, value_states, cos, sin, self.o_proj)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*):
            attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
            query_sequence_length, key_sequence_length)` if default attention is used.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    # modified: 先进行Memory部分的前向（按照memory的顺序进行，保证顺序。
    for memory_states in self.memory_detach:
        residual = hidden_states
        hidden_states = self.input_layernorm2(hidden_states)
        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
            use_cache=False,
            trigger=True,
            memory_states=memory_states
        )
        hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=True,
        trigger=False,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def compute_loss(logits, labels, shift=False):
    """
    Returns:
        token_loss: batch_size, seq_length
    """
    if shift:
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

    labels = labels.to(logits.device)
    batch_size = logits.shape[0]

    # NOTE: the loss on -100 labels is 0 by default
    token_loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), 
        labels.reshape(-1), 
        reduction="none"
    ).reshape(batch_size, -1)   # batch_size, seq_len
    
    valid_token_num = (labels != -100).sum(-1)  # batch_size
    all_valid_token_num = valid_token_num.sum()
    
    if all_valid_token_num > 0:
        loss = token_loss.sum() / valid_token_num.sum()
    else:
        loss = token_loss.sum()

    batch_loss = token_loss.sum(-1) / valid_token_num
    # prevent nan
    if (valid_token_num == 0).any():
        batch_loss = batch_loss.masked_fill(valid_token_num == 0, 0.)

    return loss, batch_loss, valid_token_num


def decoder_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

    # modified: 将输入参数从input_ids修改为input_embeds
    outputs = self.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        loss, _, valid_token_num = compute_loss(logits, labels, shift=False)
        print(f"my loss: {loss.item()}", flush=True)
        loss = loss * valid_token_num

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits)


class Teacher(Modifier):
    def __init__(self, model):
        super().__init__(model, None, None)
    
    def reset(self):
        raise NotImplementedError
    
    def ft_params(self):
        raise NotImplementedError
    


class Decoder(torch.nn.Module):
    def __init__(self, decoder, chunk_size):
        super().__init__()
        self.decoder = decoder

        for layer in decoder.model.layers:
            layer.forward = types.MethodType(decoder_layer_forward, layer)
            layer.self_attn.forward = types.MethodType(decoder_attn_forward, layer.self_attn)
            layer.self_attn.chunk_size = chunk_size
            layer.memory_detach = []
            layer.memory = []


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


    def forward(self, input_ids, labels=None):
        inputs_embeds = self.decoder.model.embed_tokens(input_ids).cpu()
        outputs = self.decoder(inputs_embeds=inputs_embeds, labels=labels)
        return outputs


class EncoderDecoder(torch.nn.Module):
    def __init__(self, decoder, chunk_size):
        super().__init__()
        self.decoder = decoder
        self.chunk_size = chunk_size


    def ft_params(self):
        return self.decoder.ft_params()
    

    def reset(self):
        # 将累积下来的input_ids和kv cache清空
        for layer in self.decoder.decoder.model.layers:
            layer.memory_detach = []
            layer.memory = []
            
            if hasattr(layer.self_attn, "k_cache"):
                del layer.self_attn.k_cache
                del layer.self_attn.v_cache
            if hasattr(layer.self_attn, "state_cache"):
                del layer.self_attn.state_cache


    def transfer_kv_cache(self):

        for layer in self.decoder.decoder.model.layers:
            
            if layer.self_attn.state_cache.shape[-2] == self.chunk_size:
                state_cache = layer.self_attn.state_cache
                del layer.self_attn.state_cache
            else:
                state_cache = layer.self_attn.state_cache[:,:self.chunk_size,:]
                layer.self_attn.state_cache = layer.self_attn.state_cache[:,self.chunk_size:,:]

            # 处理这些state cache
            state_cache_detach = state_cache.detach()
            state_cache_detach.requires_grad_(True)
            layer.memory.append(state_cache)
            layer.memory_detach.append(state_cache_detach)
            
            # 将kv cache中打头的chunk size个kv cache删除
            # if layer.self_attn.k_cache.shape[-2] == self.chunk_size:
            #     del layer.self_attn.k_cache
            #     del layer.self_attn.v_cache
            # else:
            #     layer.self_attn.k_cache = layer.self_attn.k_cache[:,:,self.chunk_size:,:]
            #     layer.self_attn.v_cache = layer.self_attn.v_cache[:,:,self.chunk_size:,:]


    def forward(
            self, 
            input_ids, 
            labels=None,
            show_debug_message=False
        ):

        assert input_ids.shape[1] <= self.chunk_size
        print = builtins.print if show_debug_message else fake_print

        print("=" * 80)
        print(f"In EncDec forward function")
        print(f"\t* input_ids: {input_ids.shape}")

        print(f"\tCurrent State:")
        print(f"\t\tlen(memory): {self.decoder.decoder.model.layers[0].memory_detach.__len__()}")
        print(f"\t\tlen(state_cache): {self.decoder.decoder.model.layers[0].self_attn.state_cache.shape[-2] if hasattr(self.decoder.decoder.model.layers[0].self_attn, 'state_cache') else 0}")
        print(f"\t\tlen(kv cache): {self.decoder.decoder.model.layers[0].self_attn.k_cache.shape[-2] if hasattr(self.decoder.decoder.model.layers[0].self_attn, 'k_cache') else 0}\n")

        print(f"\tActions:")

        outputs = self.decoder(input_ids=input_ids, labels=labels)

        print(f"\t\t{input_ids.shape[1]} tokens newly come in\n")
        print(f"\tCurrent State:")
        print(f"\t\tlen(memory): {self.decoder.decoder.model.layers[0].memory_detach.__len__()}")
        print(f"\t\tlen(state_cache): {self.decoder.decoder.model.layers[0].self_attn.state_cache.shape[-2] if hasattr(self.decoder.decoder.model.layers[0].self_attn, 'state_cache') else 0}")
        print(f"\t\tlen(kv cache): {self.decoder.decoder.model.layers[0].self_attn.k_cache.shape[-2] if hasattr(self.decoder.decoder.model.layers[0].self_attn, 'k_cache') else 0}\n")
        print(f"\tActions:")
        
        # 当input_ids超过chunk_size时，使用encoder进行压缩
        while (hasattr(self.decoder.decoder.model.layers[0].self_attn, "state_cache") 
               and self.decoder.decoder.model.layers[0].self_attn.state_cache.shape[-2] >= self.chunk_size):
            self.transfer_kv_cache()
            print(f"\t\tCompression occured!")

        print()
        print(f"\tCurrent State:")
        print(f"\t\tlen(memory): {self.decoder.decoder.model.layers[0].memory_detach.__len__()}")
        print(f"\t\tlen(state_cache): {self.decoder.decoder.model.layers[0].self_attn.state_cache.shape[-2] if hasattr(self.decoder.decoder.model.layers[0].self_attn, 'state_cache') else 0}")
        print(f"\t\tlen(kv cache): {self.decoder.decoder.model.layers[0].self_attn.k_cache.shape[-2] if hasattr(self.decoder.decoder.model.layers[0].self_attn, 'k_cache') else 0}\n")
        print("", flush=True)

        return outputs


class LlamaHierarchical(SegmentRecurrentModifier):
    def __init__(self, model, save_ckp, load_ckp, config):

        self.get_conf(config)
        chunk_size = self.conf["chunk_size"]

        kwargs = {
            "device": model.model.layers[0].self_attn.q_proj.weight.device,
            "dtype": model.model.layers[0].self_attn.q_proj.weight.dtype
        }

        for layer in model.model.layers:
            layer.self_attn.q_proj2 = torch.nn.Linear(4096, 4096, bias=False, **kwargs)
            layer.self_attn.k_proj2 = torch.nn.Linear(4096, 4096, bias=False, **kwargs)
            layer.self_attn.v_proj2 = torch.nn.Linear(4096, 4096, bias=False, **kwargs)
            layer.self_attn.o_proj2 = torch.nn.Linear(4096, 4096, bias=False, **kwargs)
            layer.input_layernorm2 = LlamaRMSNorm(4096, 1e-5)

            layer.self_attn.q_proj2.weight.data = layer.self_attn.q_proj.weight.data.clone()
            layer.self_attn.k_proj2.weight.data = layer.self_attn.k_proj.weight.data.clone()
            layer.self_attn.v_proj2.weight.data = layer.self_attn.v_proj.weight.data.clone()
            layer.self_attn.o_proj2.weight.data = layer.self_attn.o_proj.weight.data.clone()
            layer.self_attn.o_proj2.weight.data.fill_(0)

            layer.input_layernorm2.weight.data = layer.input_layernorm.weight.data.clone()

        model.forward = types.MethodType(decoder_model_forward, model)
        decoder = Decoder(model, chunk_size=chunk_size)
        encoder_decoder = EncoderDecoder(decoder, chunk_size=chunk_size)

        super().__init__(encoder_decoder, save_ckp, load_ckp, chunk_size=chunk_size)

    def ft_params(self):
        return self.model.ft_params()

    def reset(self):
        self.model.reset()

    def get_memories(self, segment_id):
        """
        当states存在但是grads不存在的时候，必须返回grads=0
        当states存在且grads同样存在的时候，正常返回
        """

        states = []
        for layer in self.model.decoder.decoder.model.layers:
            states += [
                layer.memory[segment_id].cpu()
            ]
        states = torch.cat(states, dim=0)

        if self.model.decoder.decoder.model.layers[0].memory_detach[segment_id].grad is not None:
            grads = []
            for layer in self.model.decoder.decoder.model.layers:
                grads += [
                    layer.memory_detach[segment_id].grad.data.cpu()
                ]
            grads = torch.cat(grads, dim=0)
        else:
            grads = torch.zeros_like(states)
        
        return grads, states
