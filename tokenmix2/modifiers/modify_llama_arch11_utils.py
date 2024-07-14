import torch
import math
from src.modifiers.modify_llama import check_and_apply_rope, flash_attn_func


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


def do_hidden_attn(query, key, value, cos, sin, out_proj: torch.nn.Linear = None):
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


def do_gate_attn(query, key, value, cos, sin, layer_id, o_proj):
    batch_size, num_heads, num_query, head_dim = query.shape
    Q, K, V = check_and_apply_rope(query, key, value, cos, sin)
    mask = gen_mask(num_query, Q.dtype, Q.device, layer_id=layer_id)

    score = Q @ K.transpose(-1,-2) / math.sqrt(128)
    score = score + mask
    attn = torch.softmax(score, dim=-1, dtype=torch.float32).type(score.dtype)

    output = attn @ V
    output = output.transpose(1,2).flatten(2)

    return o_proj(output)
