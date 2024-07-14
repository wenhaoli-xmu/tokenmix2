import torch
import math
from src.modifiers.modify_llama import flash_attn_func, new_posid, apply_rotary_pos_emb
from functools import partial


def check_and_apply_gate_rope(query, key, value, cos, sin):
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2]

    assert key.shape == (batch_size, num_heads, num_kv, head_dim)
    assert value.shape == (batch_size, num_heads, num_kv, head_dim)

    new_posid_spec = partial(new_posid, device=query.device, dtype=query.dtype, bsz=batch_size)
    posid = new_posid_spec(num_query)

    mem_key, hid_key, bcn_key = key.chunk(3, dim=-2)
    hid_key = apply_rotary_pos_emb(hid_key, cos, sin, posid)

    key = torch.cat([mem_key, hid_key, bcn_key], dim=-2)
    
    return query, key, value


def check_and_apply_hidden_rope(query, key, value, cos, sin):
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2]

    assert key.shape == (batch_size, num_heads, num_kv, head_dim)
    assert value.shape == (batch_size, num_heads, num_kv, head_dim)

    mem_key, hid_key = key.chunk(2, dim=-2)

    new_posid_spec = partial(new_posid, device=query.device, dtype=query.dtype, bsz=batch_size)
    posid = new_posid_spec(num_query)

    query = apply_rotary_pos_emb(query, cos, sin, posid)
    hid_key = apply_rotary_pos_emb(hid_key, cos, sin, posid)
    key = torch.cat([mem_key, hid_key], dim=-2)

    return query, key, value


def qkv_proj(states, q_proj, k_proj, v_proj):
    ques = q_proj(states).unflatten(-1, (32,128)).transpose(1,2)
    keys = k_proj(states).unflatten(-1, (32,128)).transpose(1,2)
    vals = v_proj(states).unflatten(-1, (32,128)).transpose(1,2)
    return ques, keys, vals


def do_hidden_attn(
        query, 
        key, 
        value, 
        cos, 
        sin,
        out_proj: torch.nn.Linear = None, 
    ):
    """
    仅仅支持下三角形的attention mask
    """
    batch_size, num_heads, num_query, head_dim = query.shape

    Q, K, V = check_and_apply_hidden_rope(query, key, value, cos, sin)

    # ================================
    # NOTE: test
    # if show_mem and layer_idx == show_idx:
    #     import matplotlib.pyplot as plt
    #     import random
    #     assert Q.shape[-2] * 2 == K.shape[-2]
    #     hid_que = Q
    #     mem_key, _ = K.chunk(2, dim=-2)
    #     attn = hid_que @ mem_key.transpose(-1,-2)
    #     attn = attn.softmax(dim=-1)
        
    #     plt.figure(figsize=(16,4))
    #     for i in range(4):
    #         head = random.choice(range(32))
    #         attn_head = attn[0,head,:,:].type(torch.float32).detach().cpu()
    #         plt.subplot(141 + i)
    #         plt.imshow(attn_head)
    #         plt.title(f"layer={layer_idx}, head={head}")
    #     plt.savefig("mem.jpg", dpi=960)
    #     import IPython
    #     IPython.embed(header=f"In show mem. layer={layer_idx}")
    # ================================

    Q, K, V = Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2)

    attn_output = flash_attn_func(Q, K, V, causal=True)
    attn_output = attn_output.reshape(batch_size, num_query, num_heads * head_dim).contiguous()
    
    if out_proj is not None:
        attn_output = out_proj(attn_output)

    return attn_output


def do_gate_attn(query, key, value, cos, sin, layer_id, o_proj):
    
    batch_size, num_heads, num_query, head_dim = query.shape

    Q, K, V = check_and_apply_gate_rope(query, key, value, cos, sin)
    Q, K, V = Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2)

    attn_output = flash_attn_func(Q, K, V, causal=False)
    attn_output = attn_output.reshape(batch_size, num_query, num_heads * head_dim).contiguous()
    
    if o_proj is not None:
        attn_output = o_proj(attn_output)

    return attn_output
