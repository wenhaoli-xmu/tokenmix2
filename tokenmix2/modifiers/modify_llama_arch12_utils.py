import torch
import math
from src.modifiers.modify_llama import flash_attn_func, new_posid, apply_rotary_pos_emb
from functools import partial


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

    mask[0,0,:,:num_states].fill_(0)
    mask[0,0,:,num_states:2*num_states].fill_diagonal_(0)
    mask[0,0,:,-num_states:].fill_diagonal_(0)
    mask = mask.type(dtype)

    if debug and layer_id == 0:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(mask[0,0].float().cpu().to(torch.float32))
        plt.savefig("mask.jpg", dpi=300)
        import IPython; IPython.embed(header='in generate_encoder_mask')

    return mask


def fast_hidden_attn(query, key, value, cos, sin, out_proj: torch.nn.Linear = None):

    def apply_hidden_rope(query, key, value, cos, sin):
        chunk_size = query.shape[-2]
        assert key.shape[-2] == 2 * chunk_size and key.shape == value.shape
        new_posid_spec = partial(new_posid, device=query.device, dtype=query.dtype, bsz=1)

        query_posid = new_posid_spec(chunk_size + 1)[:,-chunk_size:]
        key_posid = torch.cat([torch.zeros_like(query_posid), query_posid], dim=-1)

        Q = apply_rotary_pos_emb(query, cos, sin, query_posid)
        K = apply_rotary_pos_emb(key, cos, sin, key_posid)
        V = value

        return Q, K, V

    batch_size, num_heads, num_query, head_dim = query.shape
    Q, K, V = apply_hidden_rope(query, key, value, cos, sin)
    Q, K, V = Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2)

    attn_output = flash_attn_func(
        Q, K, V, causal=True
    )
    attn_output = attn_output.reshape(batch_size, num_query, num_heads * head_dim).contiguous()
    
    if out_proj is not None:
        attn_output = out_proj(attn_output)

    return attn_output


def do_gate_attn(query, key, value, cos, sin, layer_id, o_proj):
    chunk_size = query.shape[-2]
    assert key.shape[-2] == 3 * chunk_size and key.shape == value.shape

    def apply_gate_rope(query, key, value, cos, sin):
        chunk_size = query.shape[-2]
        new_posid_spec = partial(new_posid, device=query.device, dtype=query.dtype, bsz=1)

        query_posid = new_posid_spec(chunk_size).fill_(chunk_size + 1)
        key_posid = torch.cat([
            new_posid_spec(chunk_size),
            new_posid_spec(chunk_size).fill_(chunk_size),
            new_posid_spec(chunk_size).fill_(chunk_size + 1)
        ], dim=-1)

        Q = apply_rotary_pos_emb(query, cos, sin, query_posid)
        K = apply_rotary_pos_emb(key, cos, sin, key_posid)
        V = value

        return Q, K, V
    
    Q, K, V = apply_gate_rope(query, key, value, cos, sin)
    mask = gen_mask(chunk_size, Q.dtype, Q.device, layer_id=layer_id)

    score = Q @ K.transpose(-1,-2) / math.sqrt(128)
    score = score + mask
    attn = torch.softmax(score, dim=-1, dtype=torch.float32).type(score.dtype)

    output = attn @ V
    output = output.transpose(1,2).flatten(2)

    return o_proj(output)
    


def fast_gate_attn(query, key, value, cos, sin, o_proj):
    mem_keys, hid_keys, bcn_keys = key.chunk(3, dim=-2)
    mem_vals, hid_vals, bcn_vals = value.chunk(3, dim=-2)

    chunk_size = mem_keys.shape[-2]
    assert query.shape[-2] == chunk_size

    def apply_gate_rope(query, mem_keys, hid_keys, bcn_keys, cos, sin):
        chunk_size = query.shape[-2]
        assert query.shape == mem_keys.shape and query.shape == hid_keys.shape and query.shape == bcn_keys.shape
        new_posid_spec = partial(new_posid, device=query.device, dtype=query.dtype, bsz=1)

        query_posid = new_posid_spec(chunk_size).fill_(chunk_size + 1)
        mem_keys_posid = new_posid_spec(chunk_size).fill_(chunk_size)
        hid_keys_posid = new_posid_spec(chunk_size)
        bcn_keys_posid = new_posid_spec(chunk_size).fill_(chunk_size + 1)

        query = apply_rotary_pos_emb(query, cos, sin, query_posid)
        mem_keys = apply_rotary_pos_emb(mem_keys, cos, sin, mem_keys_posid)
        hid_keys = apply_rotary_pos_emb(hid_keys, cos, sin, hid_keys_posid)
        bcn_keys = apply_rotary_pos_emb(bcn_keys, cos, sin, bcn_keys_posid)

        return query, mem_keys, hid_keys, bcn_keys

    query, mem_keys, hid_keys, bcn_keys = apply_gate_rope(query, mem_keys, hid_keys, bcn_keys, cos, sin)

    hid_scores = query @ hid_keys.transpose(-1,-2) / math.sqrt(128)
    mem_scores = (query * mem_keys).sum(-1, keepdim=True) / math.sqrt(128)
    bcn_scores = (query * bcn_keys).sum(-1, keepdim=True) / math.sqrt(128)

    scores = torch.cat([hid_scores, mem_scores, bcn_scores], dim=-1)
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32).type(scores.dtype)

    hid_attn = attn[:,:,:,:chunk_size]
    mem_attn = attn[:,:,:,chunk_size:chunk_size+1]
    bcn_attn = attn[:,:,:,-1:]
    assert hid_attn.shape[-1] + mem_attn.shape[-1] + bcn_attn.shape[-1] == chunk_size + 2

    hid_outs = hid_attn @ hid_vals
    mem_outs = mem_attn * mem_vals
    bcn_outs = bcn_attn * bcn_vals
    outs = hid_outs + mem_outs + bcn_outs
    outs = outs.transpose(1,2).flatten(2)

    return o_proj(outs)
