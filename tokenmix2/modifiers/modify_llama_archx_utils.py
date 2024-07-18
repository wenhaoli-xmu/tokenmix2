import torch
import math
from ..modifiers.modify_llama import flash_attn_func, new_posid, apply_rotary_pos_emb, check_and_apply_rope
from functools import partial


def check_and_apply_expand_rope(query, key, value, cos, sin):
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2]

    assert key.shape == (batch_size, num_heads, num_kv, head_dim)
    assert value.shape == (batch_size, num_heads, num_kv, head_dim)

    new_posid_spec = partial(new_posid, device=query.device, dtype=query.dtype, bsz=batch_size)

    Q = apply_rotary_pos_emb(
        mat=query, 
        cos=cos, 
        sin=sin, 
        position_ids=new_posid_spec(num_query))
    
    K = apply_rotary_pos_emb(
        mat=key, 
        cos=cos, 
        sin=sin, 
        position_ids=torch.cat([new_posid_spec(num_query), new_posid_spec(num_query), new_posid_spec(num_query)], dim=-1))
    
    V = value

    return Q, K, V


def qkv_proj(states, q_proj, k_proj, v_proj):
    ques = q_proj(states).unflatten(-1, (32,128)).transpose(1,2)
    keys = k_proj(states).unflatten(-1, (32,128)).transpose(1,2)
    vals = v_proj(states).unflatten(-1, (32,128)).transpose(1,2)
    return ques, keys, vals


def gen_mask(num_states, dtype, device, layer_id, debug=False):
    mask = torch.full(
        (1, 1, num_states, 3 * num_states), 
        torch.finfo(dtype).min, 
        dtype=torch.float32, 
        device=device
    )

    mask[0,0,:,:num_states].fill_diagonal_(0)
    mask[0,0,:,num_states:2*num_states].triu_(diagonal=1)
    mask[0,0,:,-num_states:].fill_diagonal_(0)
    mask = mask.type(dtype)

    if debug and layer_id == 0:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(mask[0,0].float().cpu().to(torch.float32))
        plt.savefig("mask.jpg", dpi=300)
        import IPython; IPython.embed(header='in generate_encoder_mask')

    return mask


def gen_mask_for_fast_gate_attn(num_states, dtype, device, layer_id, debug=False):
    mask = torch.full(
        (1, 1, num_states, 2 + num_states), 
        torch.finfo(dtype).min, 
        dtype=torch.float32, 
        device=device
    )

    mask[0,0,:,:1].fill_(0)
    mask[0,0,:,1:-1].triu_(diagonal=1)
    mask[0,0,:,-1:].fill_(0)
    mask = mask.type(dtype)

    if debug and layer_id == 0:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(mask[0,0].float().cpu().to(torch.float32))
        plt.savefig("mask.jpg", dpi=300)
        import IPython; IPython.embed(header='in generate_encoder_mask')

    return mask


def check_and_apply_hidden_rope(query, key, value, cos, sin):
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2]

    assert key.shape == (batch_size, num_heads, num_kv, head_dim)
    assert value.shape == (batch_size, num_heads, num_kv, head_dim)

    new_posid_spec = partial(new_posid, device=query.device, dtype=query.dtype, bsz=batch_size)

    posid = new_posid_spec(num_query)
    Q = apply_rotary_pos_emb(query, cos, sin, posid)
    K = apply_rotary_pos_emb(key, cos, sin, torch.cat([posid, posid], dim=-1))
    V = value

    return Q, K, V


def do_hidden_attn(
        query, 
        key, 
        value, 
        cos, 
        sin, 
        layer_idx, 
        use_expand: bool = False, 
        out_proj: torch.nn.Linear = None, 
    ):
    """
    仅仅支持下三角形的attention mask
    """
    batch_size, num_heads, num_query, head_dim = query.shape

    if use_expand:
        Q, K, V = check_and_apply_hidden_rope(query, key, value, cos, sin)
    else:
        Q, K, V = check_and_apply_rope(query, key, value, cos, sin)

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

    attn_output = flash_attn_func(
        Q, K, V, causal=True
    )
    attn_output = attn_output.reshape(batch_size, num_query, num_heads * head_dim).contiguous()
    
    if out_proj is not None:
        attn_output = out_proj(attn_output)

    return attn_output


def fast_gate_attn(query, key, value, cos, sin, layer_id, o_proj, use_expand=False):
    batch_size, num_heads, num_query, head_dim = query.shape

    if use_expand:
        Q, K, V = check_and_apply_expand_rope(query, key, value, cos, sin)
    else:
        Q, K, V = check_and_apply_rope(query, key, value, cos, sin)

    # =======================
    # NOTE: test
    # Q = Q.type(torch.float32)
    # K = K.type(torch.float32)
    # V = V.type(torch.float32)
    # =======================

    bcn_que = Q
    mem_key, hid_key, bcn_key = K.chunk(3, dim=-2)
    mem_val, hid_val, bcn_val = V.chunk(3, dim=-2)

    mem_score = (bcn_que * mem_key).sum(dim=-1, keepdim=True) / math.sqrt(128)
    hid_score = bcn_que @ hid_key.transpose(-1,-2) / math.sqrt(128)
    bcn_score = (bcn_que * bcn_key).sum(dim=-1, keepdim=True) / math.sqrt(128)

    cat_score = torch.cat([mem_score, hid_score, bcn_score], dim=-1)
    cat_score = cat_score + gen_mask_for_fast_gate_attn(num_query, bcn_que.dtype, bcn_que.device, layer_id)
    cat_attn = torch.softmax(cat_score, dim=-1, dtype=torch.float32).to(mem_score.dtype)

    mem_attn = cat_attn[...,:1]
    hid_attn = cat_attn[...,1:-1]
    bcn_attn = cat_attn[...,-1:]

    # ===================
    # # NOTE: test
    # score = Q @ K.transpose(-1,-2) / math.sqrt(128)
    # score = score + gen_mask(num_states=num_query, dtype=Q.dtype, device=Q.device, layer_id=layer_id)
    # attn = score.softmax(dim=-1, dtype=torch.float32).to(Q.dtype)
    # mem_attn2, hid_attn2, bcn_attn2 = attn.chunk(3, dim=-1)
    # mem_attn2 = mem_attn2.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
    # bcn_attn2 = bcn_attn2.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
    # print(torch.dist(mem_attn, mem_attn2))
    # print(torch.dist(hid_attn, hid_attn2))
    # print(torch.dist(bcn_attn, bcn_attn2))
    # import IPython
    # IPython.embed(header='debug')
    # ===================

    mem_out = mem_attn * mem_val
    hid_out = hid_attn @ hid_val
    bcn_out = bcn_attn * bcn_val

    out = mem_out + hid_out + bcn_out
    out = out.transpose(1,2).flatten(2)

    return o_proj(out)


def do_beacon_attn(query, key, value, cos, sin, layer_id, o_proj, use_expand=False):
    
    # ===============================
    # NOTE: test
    # query = query.type(torch.float32)
    # key = key.type(torch.float32)
    # value = value.type(torch.float32)
    # cos = cos.type(torch.float32)
    # sin = sin.type(torch.float32)
    # o_proj.weight.data = o_proj.weight.data.type(torch.float32)
    # ===============================

    batch_size, num_heads, num_query, head_dim = query.shape

    if use_expand:
        Q, K, V = check_and_apply_expand_rope(query, key, value, cos, sin)
    else:
        Q, K, V = check_and_apply_rope(query, key, value, cos, sin)

    mask = gen_mask(num_query, Q.dtype, Q.device, layer_id=layer_id)

    score = Q @ K.transpose(-1,-2) / math.sqrt(128)
    score = score + mask
    attn = torch.softmax(score, dim=-1, dtype=torch.float32).type(score.dtype)

    output = attn @ V
    output = output.transpose(1,2).flatten(2)

    result = o_proj(output)

    # ===================================
    # NOTE: test
    # result2 = fast_gate_attn(query, key, value, cos, sin, layer_id, o_proj, use_expand)
    # print(torch.dist(result, result2))
    # import IPython
    # IPython.embed(header='检查fast_gate_attn和do_beacon_attn的区别')
    # ===================================

    return result


def do_forget_attn(query, key, value, cos, sin, layer_id, o_proj, use_expand=False):
    """
    仅仅支持下三角形的attention mask
    """
    batch_size, num_heads, num_query, head_dim = query.shape

    if use_expand:
        Q, K, V = check_and_apply_expand_rope(query, key, value, cos, sin)
    else:
        Q, K, V = check_and_apply_rope(query, key, value, cos, sin)

    mask = gen_mask(num_query, Q.dtype, Q.device, layer_id=layer_id)

    score = Q @ K.transpose(-1,-2) / math.sqrt(128)
    score = score + mask
    attn = torch.softmax(score, dim=-1, dtype=torch.float32).type(score.dtype)

    output = attn @ V
    output = output.transpose(1,2).flatten(2)

    return o_proj(output)