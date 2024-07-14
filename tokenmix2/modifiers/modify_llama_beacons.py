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
    repeat_kv
    )
from transformers.cache_utils import Cache

from transformers.cache_utils import Cache
import torch.nn.functional as F

from src.modifier import Modifier, SegmentRecurrentModifier
from functools import partial

from flash_attn import flash_attn_func
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from copy import deepcopy
import builtins, random


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


def check_and_apply_encoder_rope(query, key, value, cos, sin, num_ordinal, num_memory, num_beacons):
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2]


    assert key.shape == (batch_size, num_heads, num_kv, head_dim)
    assert value.shape == (batch_size, num_heads, num_kv, head_dim)

    assert num_query == num_ordinal + num_memory
    assert num_kv == num_ordinal + num_memory + num_beacons

    new_posid_spec = partial(new_posid, device=query.device, dtype=query.dtype, bsz=batch_size)

    if num_memory > 0:
        ordinal_query = apply_rotary_pos_emb(query[:,:,:-num_memory,:], cos, sin, new_posid_spec(num_ordinal) + num_beacons)
        ordinal_key = apply_rotary_pos_emb(key[:,:,:-num_memory,:], cos, sin, new_posid_spec(num_beacons + num_ordinal))
        cover_tokens = num_ordinal // num_memory
        memory_query = apply_rotary_pos_emb(query[:,:,-num_memory:,:], cos, sin, (new_posid_spec(num_memory) + 1) * cover_tokens + num_beacons)
        memory_key = apply_rotary_pos_emb(key[:,:,-num_memory:,:], cos, sin, (new_posid_spec(num_memory) + 1) * cover_tokens + num_beacons)
        Q = torch.cat([ordinal_query, memory_query], dim=-2)
        K = torch.cat([ordinal_key, memory_key], dim=-2)
    else:
        Q = apply_rotary_pos_emb(query, cos, sin, new_posid_spec(num_ordinal) + num_beacons)
        K = apply_rotary_pos_emb(key, cos, sin, new_posid_spec(num_beacons + num_ordinal))

    V = value

    return Q, K, V


def generate_encoder_mask(num_ordinal, num_memory, num_beacons, dtype, device, layer_id, debug=False):
    mask = torch.full(
        (1, 1, num_ordinal + num_memory, num_beacons + num_ordinal + num_memory), 
        torch.finfo(dtype).min, 
        dtype=torch.float32, 
        device=device
    )

    mask[0,0,:,:num_beacons].fill_(0)
    mask[0,0,:num_ordinal,num_beacons:num_ordinal+num_beacons].triu_(diagonal=1)
    mask[0,0,num_ordinal:,num_beacons+num_ordinal:].fill_diagonal_(0)
    mask = mask.type(dtype)

    mask[0,0,num_ordinal:,num_beacons:num_beacons+num_ordinal].fill_(0)
    for i in range(num_memory):
        start = (i + 1) * (num_ordinal // num_memory) + num_beacons
        end = num_ordinal + num_beacons
        mask[0,0,num_ordinal+i, start: end] = torch.finfo(dtype).min

    if debug and layer_id == 0:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(mask[0,0].float().cpu().to(torch.float32))
        plt.savefig("mask.jpg", dpi=300)
        import IPython; IPython.embed(header='in generate_encoder_mask')

    return mask


def do_encoder_attn(query, key, value, cos, sin, beacon_o_proj, o_proj, num_ordinal, num_memory, num_beacons, layer_id):
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2]
    
    Q, K, V = check_and_apply_encoder_rope(query, key, value, cos, sin, num_ordinal, num_memory, num_beacons)

    mask = generate_encoder_mask(num_ordinal, num_memory, num_beacons, dtype=query.dtype, device=query.device, layer_id=layer_id)

    score = Q @ K.transpose(-1,-2) / math.sqrt(128)
    score = score + mask
    attn = torch.softmax(score, dim=-1, dtype=torch.float32).type(score.dtype)

    output = attn @ V
    output = output.transpose(1,2).flatten(2)

    ordinal_result = o_proj(output[:,:-num_memory,:])
    beacon_result = beacon_o_proj(output[:,-num_memory:,:])

    output = torch.cat([ordinal_result, beacon_result], dim=-2)

    return output


def attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len = hidden_states.shape[:2]
        num_ordinal = q_len - self.num_memory

        # 将ordinal和beacons分割，并通过各自的proj矩阵
        beacon_states = hidden_states[:,-self.num_memory:,:]
        ordinal_states = hidden_states[:,:-self.num_memory,:]

        beacon_query = self.beacon_q_proj(beacon_states)
        beacon_key = self.beacon_k_proj(beacon_states)
        beacon_value = self.beacon_v_proj(beacon_states)
        ordinal_query = self.q_proj(ordinal_states)
        ordinal_key = self.k_proj(ordinal_states)
        ordinal_value = self.v_proj(ordinal_states)

        num_beacons = sum([mem.shape[-2] for mem in self.memory_key_detach])

        # 将memory, ordinal, beacon三者拼接起来
        query_states = torch.cat([ordinal_query, beacon_query], dim=-2)
        key_states = torch.cat([*self.memory_key_detach, ordinal_key, beacon_key], dim=-2)
        value_states = torch.cat([*self.memory_value_detach, ordinal_value, beacon_value], dim=-2)

        # 将得到的beacon的kv cache缓存起来
        if beacon_key.shape[-2] > 0:
            self.memory_key.append(beacon_key)
            self.memory_key_detach.append(beacon_key.detach())
            self.memory_key_detach[-1].requires_grad_(True)
            
            self.memory_value.append(beacon_value)
            self.memory_value_detach.append(beacon_value.detach())
            self.memory_value_detach[-1].requires_grad_(True)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # modified: 将这里的seq_len=kv_seq_len修改为2048
        cos, sin = self.rotary_emb(value_states, seq_len=4096)

        attn_output = do_encoder_attn(
            query_states, 
            key_states, 
            value_states, 
            cos, sin,
            self.beacon_o_proj, self.o_proj, 
            num_ordinal, self.num_memory, num_beacons,
            self.layer_idx)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def layer_forward(
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

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
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


def model_forward(
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

    # modified: 新增加
    inputs_embeds = self.model.embed_tokens(input_ids)
    assert input_ids.shape[-1] <= self.chunk_size
    if input_ids.shape[-1] == self.chunk_size:
        compress = random.choice([2,4,8,16,32,64,128])
        num_memory = input_ids.shape[-1] // compress
        beacon_ids = torch.zeros((1, num_memory), dtype=torch.long)
        beacon_embeds = self.model.beacon_embed_tokens(beacon_ids)
        inputs_embeds = torch.cat([inputs_embeds, beacon_embeds], dim=-2)
    else:
        num_memory = 0
    for layer in self.model.layers:
        layer.self_attn.num_memory = num_memory

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

    assert labels.ndim == 2 and labels.shape[0] == 1
    labels = torch.cat([labels, labels.new_full((1, num_memory), -100, dtype=torch.long)], dim=-1)

    loss = None
    if labels is not None:
        loss, _, valid_token_num = compute_loss(logits, labels, shift=False)
        print(f"my loss: {loss.item()}")
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


class LlamaBeacons(SegmentRecurrentModifier):
    def __init__(self, model, save_ckp, load_ckp, config):

        self.get_conf(config)

        self.chunk_size = self.conf['chunk_size']

        kwargs = {
            "device": model.model.layers[0].self_attn.q_proj.weight.device,
            "dtype": model.model.layers[0].self_attn.q_proj.weight.dtype
        }

        # 设置beacon embedding
        model.model.beacon_embed_tokens = nn.Embedding(1, 4096, **kwargs)
        model.model.beacon_embed_tokens.weight.data[:] = model.model.embed_tokens.weight[2,:].data

        # 模型前向传播
        model.forward = types.MethodType(model_forward, model)
        model.chunk_size = self.chunk_size

        for layer in model.model.layers:

            layer.self_attn.beacon_q_proj = torch.nn.Linear(4096, 4096, bias=False, **kwargs)
            layer.self_attn.beacon_k_proj = torch.nn.Linear(4096, 4096, bias=False, **kwargs)
            layer.self_attn.beacon_v_proj = torch.nn.Linear(4096, 4096, bias=False, **kwargs)
            layer.self_attn.beacon_o_proj = torch.nn.Linear(4096, 4096, bias=False, **kwargs)

            layer.self_attn.beacon_q_proj.weight.data[:] = layer.self_attn.q_proj.weight.data
            layer.self_attn.beacon_k_proj.weight.data[:] = layer.self_attn.k_proj.weight.data
            layer.self_attn.beacon_v_proj.weight.data[:] = layer.self_attn.v_proj.weight.data
            layer.self_attn.beacon_o_proj.weight.data[:] = layer.self_attn.o_proj.weight.data

            layer.self_attn.memory_key = []
            layer.self_attn.memory_value = []
            layer.self_attn.memory_key_detach = []
            layer.self_attn.memory_value_detach = []

            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(attn_forward, layer.self_attn)

        super().__init__(model, save_ckp, load_ckp, chunk_size=self.chunk_size)

    def ft_params(self):
        params = [self.model.model.beacon_embed_tokens.weight]
        for layer in self.model.model.layers:
            params += [
                layer.self_attn.beacon_q_proj.weight,
                layer.self_attn.beacon_k_proj.weight,
                layer.self_attn.beacon_v_proj.weight,
                layer.self_attn.beacon_o_proj.weight
            ]
        return params

    def reset(self):
        for layer in self.model.model.layers:
            layer.self_attn.memory_key = []
            layer.self_attn.memory_value = []
            layer.self_attn.memory_key_detach = []
            layer.self_attn.memory_value_detach = []

    def get_memories(self, segment_id):
        """
        当states存在但是grads不存在的时候，必须返回grads=0
        当states存在且grads同样存在的时候，正常返回
        """
        states = []
        for layer in self.model.model.layers:
            states += [
                layer.self_attn.memory_key[segment_id].cpu(),
                layer.self_attn.memory_value[segment_id].cpu()
            ]
        states = torch.cat(states, dim=0)

        if self.model.model.layers[0].self_attn.memory_key_detach[segment_id].grad is not None:
            grads = []
            for layer in self.model.model.layers:
                grads += [
                    layer.self_attn.memory_key_detach[segment_id].grad.data.cpu(),
                    layer.self_attn.memory_value_detach[segment_id].grad.data.cpu()
                ]
            grads = torch.cat(grads, dim=0)
        else:
            grads = torch.zeros_like(states)

        return grads, states
