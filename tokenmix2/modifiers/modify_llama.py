import math
import torch
import torch.utils.checkpoint
from transformers.models.llama.modeling_llama import rotate_half
from functools import partial
from flash_attn import flash_attn_func


def segment(tensor, dim, n):
    total_length = tensor.shape[dim]

    for start in range(0, total_length, n):
        end = min(start + n, total_length)
        indices = [slice(None)] * tensor.ndim
        indices[dim] = slice(start, end)
        yield tensor[tuple(indices)]


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


def check_and_apply_rope_hybird(query, key, value, cos, sin):
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2] - num_query

    assert key.shape == (batch_size, num_heads, num_query + num_kv, head_dim)
    assert value.shape == (batch_size, num_heads, num_query + num_kv, head_dim)

    new_posid_spec = partial(new_posid, device=query.device, dtype=query.dtype, bsz=batch_size)

    query_posid = new_posid_spec(num_query)
    key_posid = torch.cat([new_posid_spec(num_kv), new_posid_spec(num_query)], dim=-1)

    assert torch.max(query_posid).item() <= cos.shape[0]
    assert torch.max(key_posid).item() <= cos.shape[0]

    Q = apply_rotary_pos_emb(query, cos, sin, query_posid)
    K = apply_rotary_pos_emb(key, cos, sin, key_posid)
    V = value

    return Q, K, V


def generate_decoder_mask(num_querys, num_keys, dtype, device, debug=False):
    assert num_querys <= num_keys
    mask = torch.full((1,1,num_querys,num_querys), torch.finfo(dtype).min, device=device, dtype=torch.float32).triu(diagonal=1).type(dtype)
    prefix = torch.zeros((1,1,num_querys,num_keys-num_querys), device=device, dtype=dtype)
    mask = torch.cat([prefix, mask], dim=-1)

    assert mask.shape == (1, 1, num_querys, num_keys)

    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(mask[0,0].cpu())
        plt.savefig("mask.jpg", dpi=300)
        import IPython; IPython.embed(header='In generate_decoder_mask')

    assert (mask != 0).sum().item() == num_querys * (num_querys - 1) / 2
    assert (mask == 0).sum().item() == num_querys * num_keys - num_querys * (num_querys - 1) / 2

    return mask


def check_and_apply_beacon_rope(query, key, value, cos, sin, num_ordinal, num_memory, num_beacons):
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


def generate_mask(num_query, num_kv, dtype, device):
    mask = torch.full(
        (1, 1, num_query, num_kv), 
        torch.finfo(dtype).min, 
        dtype=torch.float32, 
        device=device
    )
    assert num_query <= num_kv
    mask[0,0,:,-num_query:].triu_(diagonal=1)
    mask[0,0,:,:-num_query].fill_(0)
    mask = mask.type(dtype)
    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(mask[0,0].float().cpu().to(torch.float32))
        plt.savefig("mask.jpg", dpi=300)
        import IPython; IPython.embed(header='in generate_encoder_mask')
    return mask


def generate_hybird_mask(num_query, num_kv, chunk_size, dtype, device, debug=False):
   
    mask = torch.full(
        (1, 1, num_query, num_query + num_kv),
        torch.finfo(dtype).min,
        dtype=torch.float32,
        device=device)

    num_query_chunks = math.ceil(num_query / chunk_size)

    for i in range(num_query_chunks - 1):
        que_beg = (i + 1) * chunk_size
        que_end = min((i + 2) * chunk_size, num_query)
        key_beg = 0
        key_end = (i + 1) * chunk_size
        mask[0,0,que_beg:que_end,key_beg:key_end].fill_(0)

    for i in range(num_query_chunks):
        que_beg = i * chunk_size
        que_end = min((i + 1) * chunk_size, num_query)
        key_beg = i * chunk_size + num_kv
        key_end = min((i + 1) * chunk_size, num_query) + num_kv
        mask[0,0,que_beg:que_end,key_beg:key_end].triu_(diagonal=1)

    mask = mask.type(dtype)
    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(mask[0,0].float().cpu().to(torch.float32))
        plt.savefig("mask.jpg", dpi=300)
        import IPython; IPython.embed(header='in generate_encoder_mask')

    return mask


def generate_shift_mask(num_query, num_kv, dtype, device, shift_mask=True, debug=False):
    mask = torch.full(
        (1, 1, num_query, num_kv),
        torch.finfo(dtype).min,
        dtype=torch.float32,
        device=device)

    if shift_mask:
        mask[0,0,:,:].triu_(diagonal=0)
        mask[0,0,0,0].fill_(0)
    else:
        mask[0,0,:,:].triu_(diagonal=1)
    
    mask = mask.type(dtype)

    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(mask[0,0].float().cpu().to(torch.float32))
        plt.savefig("mask.jpg", dpi=300)
        import IPython; IPython.embed(header='in generate_encoder_mask')

    return mask


def generate_highlv_mask(num_query, num_kv, dtype, device, expand_type="stepwise", debug=False):
    mask = torch.full(
        (1, 1, num_query, num_kv),
        torch.finfo(dtype).min,
        dtype=torch.float32,
        device=device)

    mask[0,0,:,-num_query:].triu_(diagonal=1)

    chunk_size = (num_kv - num_query) // num_query
    
    for i in range(num_query):

        if expand_type == 'segment':
            start = i * chunk_size
            end = start + chunk_size
        elif expand_type == 'stepwise':
            start = 0
            end = (i + 1) * chunk_size
        else:
            raise NotImplementedError

        mask[0,0,i,start:end].fill_(0)

    mask = mask.type(dtype)
    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(mask[0,0].float().cpu().to(torch.float32))
        plt.savefig("mask.jpg", dpi=300)
        import IPython; IPython.embed(header='in generate_encoder_mask')

    return mask


def generate_beacon_mask(num_ordinal, num_memory, num_beacons, dtype, device, layer_id, memory_mask, debug=False):
    mask = torch.full(
        (1, 1, num_ordinal + num_memory, num_beacons + num_ordinal + num_memory), 
        torch.finfo(dtype).min, 
        dtype=torch.float32, 
        device=device
    )

    mask[0,0,:num_ordinal,:num_beacons].fill_(0)

    if memory_mask == "triu":
        mask[0,0,num_ordinal:,:num_beacons].triu_(diagonal=1)
        mask[0,0,num_ordinal:,num_beacons:num_beacons+num_ordinal].triu_(diagonal=1)
    elif memory_mask == "diag":
        mask[0,0,num_ordinal:,:num_beacons].fill_diagonal_(0)
        mask[0,0,num_ordinal:,num_beacons:num_beacons+num_ordinal].triu_(diagonal=1)
    elif memory_mask == "full":
        mask[0,0,num_ordinal:,:num_beacons].fill_(0)
        mask[0,0,num_ordinal:,num_beacons:num_beacons+num_ordinal].triu_(diagonal=1)
    elif memory_mask == "mixed":
        for i in range(num_memory):
            start = 0
            end = i * 2 + 2
            mask[0,0,num_ordinal+i,start:end].fill_(0)
    else:
        raise NotImplementedError()
    
    mask[0,0,:num_ordinal,num_beacons:num_beacons+num_ordinal].triu_(diagonal=1)
    mask[0,0,num_ordinal:,num_beacons+num_ordinal:].fill_diagonal_(0)

    mask = mask.type(dtype)

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


def do_beacon_attn(query, key, value, cos, sin, o_proj, num_ordinal, num_memory, num_beacons, layer_id, memory_mask):
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2]
    
    Q, K, V = check_and_apply_beacon_rope(query, key, value, cos, sin, num_ordinal, num_memory, num_beacons)

    mask = generate_beacon_mask(num_ordinal, num_memory, num_beacons, dtype=query.dtype, device=query.device, layer_id=layer_id, memory_mask=memory_mask)

    score = Q @ K.transpose(-1,-2) / math.sqrt(128)
    score = score + mask
    attn = torch.softmax(score, dim=-1, dtype=torch.float32).type(score.dtype)

    output = attn @ V
    output = output.transpose(1,2).flatten(2)

    return o_proj(output)


def do_causal_attn(query, key, value, cos, sin, out_proj = None):
    batch_size, num_heads, num_query, head_dim = query.shape
    query, key, value = check_and_apply_rope(query, key, value, cos, sin)

    attn_score = query @ key.transpose(-1,-2) / query.shape[-1]
    attn_mask = generate_mask(query.shape[-2], key.shape[-2], query.dtype, query.device)
    attn_score = torch.softmax(attn_score + attn_mask, dim=-1, dtype=torch.float32).type(query.dtype)
    attn_output = attn_score @ value

    attn_output = attn_output.transpose(1,2).flatten(2)

    if out_proj is not None:
        attn_output = out_proj(attn_output)

    return attn_output


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


def do_shift_mask_attn(query, key, value, cos, sin, out_proj: torch.nn.Linear, shift_mask: bool):
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2]
    Q, K, V = check_and_apply_rope(query, key, value, cos, sin)

    mask = generate_shift_mask(
        num_query,
        num_kv,
        dtype=query.dtype,
        device=query.device,
        shift_mask=shift_mask)
    
    score = Q @ K.transpose(-1,-2) / math.sqrt(128)
    score = score + mask
    attn = torch.softmax(score, dim=-1, dtype=torch.float32).type(score.dtype)

    output = attn @ V
    output = output.transpose(1,2).flatten(2)

    return out_proj(output)


def pad_merge_token(MQ, MK, MV, chunk_size):
    num_padding = chunk_size - MQ.shape[-2]
    shape_padding = list(MQ.shape)
    shape_padding[-2] = num_padding
    config = {"dtype": MQ.dtype, "device": MQ.device}

    MQ = torch.cat([MQ, torch.zeros(shape_padding, **config)], dim=-2)
    MK = torch.cat([MK, torch.zeros(shape_padding, **config)], dim=-2)
    MV = torch.cat([MV, torch.zeros(shape_padding, **config)], dim=-2)

    return MQ, MK, MV, num_padding



def do_prefill_accelerate_sdpa_attn(query, key, value, cos, sin, out_proj):
    Q, K, V = check_and_apply_rope(query, key, value, cos, sin)

    ordinal_mask = generate_mask(Q.shape[-2], K.shape[-2], Q.dtype, Q.device)
    shift_mask = generate_shift_mask(Q.shape[-2], K.shape[-2], Q.dtype, Q.device, True)
    concat_mask = torch.cat([ordinal_mask.expand(Q.shape[0] - 1, -1, -1, -1), shift_mask], dim=0)

    output = torch.nn.functional.scaled_dot_product_attention(
        query=Q,
        key=K,
        value=V,
        attn_mask=concat_mask,
        is_causal=False)
    
    output = output.transpose(1,2).flatten(2)
    output = out_proj(output)
    return output
    



def do_hybird_attn(query, key, value, cos, sin, out_proj: torch.nn.Linear, chunk_size: int):
    """
    仅仅支持下三角形的attention mask
    """
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2] - num_query
    Q, K, V = check_and_apply_rope_hybird(query, key, value, cos, sin)

    mask = generate_hybird_mask(
        num_query, 
        num_kv, 
        chunk_size, 
        dtype=query.dtype, 
        device=query.device)

    # =========================================================================================
    # NOTE: memory efficient implementation
    # outputs = []
    # for chunk_Q, chunk_mask in zip(segment(Q, dim=-2, n=128), segment(mask, dim=-2, n=128)):
    #     score = chunk_Q @ K.transpose(-1,-2) / math.sqrt(128)
    #     score = score + chunk_mask
    #     attn = torch.softmax(score, dim=-1, dtype=torch.float32).type(score.dtype)
    #     output = attn @ V
    #     output = output.transpose(1,2).flatten(2)
    #     outputs.append(output)
    # output = torch.cat(outputs, dim=-2)
    # =========================================================================================

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query=Q,
        key=K,
        value=V,
        attn_mask=mask)

    attn_output = attn_output.transpose(1,2).flatten(2)

    return out_proj(attn_output)


def do_highlv_attn(query, key, value, cos, sin, out_proj=None, expand_type='stepwise'):
    num_query = query.shape[-2]
    num_kv = key.shape[-2]
    Q, K, V = check_and_apply_rope(query, key, value, cos, sin)
    mask = generate_highlv_mask(num_query, num_kv, query.dtype, query.device, expand_type)

    score = Q @ K.transpose(-1,-2) / math.sqrt(128)
    score = score + mask
    attn = torch.softmax(score, dim=-1, dtype=torch.float32).type(score.dtype)

    output = attn @ V
    output = output.transpose(1,2).flatten(2)

    return out_proj(output) if out_proj is not None else output



def do_full_flash_attn(query, key, value, cos, sin, out_proj: torch.nn.Linear = None):
    """
    仅仅支持下三角形的attention mask
    """
    batch_size, num_heads, num_query, head_dim = query.shape
    Q, K, V = check_and_apply_rope(query, key, value, cos, sin)
    Q, K, V = Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2)

    attn_output = flash_attn_func(
        Q, K, V, causal=False
    )
    attn_output = attn_output.reshape(batch_size, num_query, num_heads * head_dim).contiguous()
    
    if out_proj is not None:
        attn_output = out_proj(attn_output)

    return attn_output


def do_causal_flash_attn_without_rope(query, key, value, out_proj: torch.nn.Linear = None):
    """
    仅仅支持下三角形的attention mask
    """
    batch_size, num_heads, num_query, head_dim = query.shape
    Q, K, V = query.transpose(1,2), key.transpose(1,2), value.transpose(1,2)

    attn_output = flash_attn_func(
        Q, K, V, causal=True
    )
    attn_output = attn_output.reshape(batch_size, num_query, num_heads * head_dim).contiguous()
    
    if out_proj is not None:
        attn_output = out_proj(attn_output)

    return attn_output


def do_full_flash_attn_without_rope(query, key, value, out_proj: torch.nn.Linear = None):
    """
    仅仅支持下三角形的attention mask
    """
    batch_size, num_heads, num_query, head_dim = query.shape
    Q, K, V = query.transpose(1,2), key.transpose(1,2), value.transpose(1,2)

    attn_output = flash_attn_func(
        Q, K, V, causal=False
    )
    attn_output = attn_output.reshape(batch_size, num_query, num_heads * head_dim).contiguous()
    
    if out_proj is not None:
        attn_output = out_proj(attn_output)

    return attn_output


def do_adapter_attn(query, key, value, out_proj: torch.nn.Linear = None):
    """
    仅仅支持下三角形的attention mask
    """
    batch_size, num_heads, num_query, head_dim = query.shape
    Q, K, V = query.transpose(1,2), key.transpose(1,2), value.transpose(1,2)

    attn_output = flash_attn_func(
        Q, K, V, causal=False
    )
    attn_output = attn_output.reshape(batch_size, num_query, num_heads * head_dim).contiguous()
    
    if out_proj is not None:
        attn_output = out_proj(attn_output)

    return attn_output


class Adapter(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        kwargs = {
            "device": layer.self_attn.q_proj.weight.data.device,
            "dtype": layer.self_attn.q_proj.weight.data.dtype,
        }
        self.adapter = torch.nn.Parameter(torch.randn((1,4,4096), **kwargs) * 1e-3, requires_grad=True) 


class ProjectHead(torch.nn.Module):
    def __init__(self, layer, zero_init=False):
        super().__init__()
        kwargs = {
            "device": layer.self_attn.q_proj.weight.data.device,
            "dtype": layer.self_attn.q_proj.weight.data.dtype,
        }
        self.key_proj = torch.nn.Linear(4096, 4096, bias=False, **kwargs)
        self.val_proj = torch.nn.Linear(4096, 4096, bias=False, **kwargs)

        self.key_proj.weight.data = layer.self_attn.k_proj.weight.data.clone()
        self.val_proj.weight.data = layer.self_attn.v_proj.weight.data.clone()

        if zero_init:
            self.key_proj.weight.data.fill_(0)
            self.val_proj.weight.data.fill_(0)

    
    def get_lora_parameters(self):
        return [
            self.key_proj.lora_A.default.weight,
            self.key_proj.lora_B.default.weight,
            self.val_proj.lora_A.default.weight,
            self.val_proj.lora_B.default.weight
        ]


    def forward(self, activation: torch.Tensor):
        cache_k = self.key_proj(activation).unflatten(-1, (32, 128)).transpose(1,2)
        cache_v = self.val_proj(activation).unflatten(-1, (32, 128)).transpose(1,2)
        return cache_k, cache_v
    

class CrossAttnQKVProj(torch.nn.Module):
    def __init__(self, layer, random_init=False, embed_dim: int = 4096):
        super().__init__()
        kwargs = {
            "device": layer.self_attn.q_proj.weight.data.device,
            "dtype": layer.self_attn.q_proj.weight.data.dtype,
        }
        self.que_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False, **kwargs)
        self.key_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False, **kwargs)
        self.val_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False, **kwargs)

        self.que_proj.weight.data = layer.self_attn.q_proj.weight.data.clone()
        self.key_proj.weight.data = layer.self_attn.k_proj.weight.data.clone()
        self.val_proj.weight.data = layer.self_attn.v_proj.weight.data.clone()

        if random_init:
            que_std = torch.std(self.que_proj.weight.data)
            key_std = torch.std(self.key_proj.weight.data)
            val_std = torch.std(self.val_proj.weight.data)
            self.que_proj.weight.data = torch.randn_like(self.que_proj.weight.data) * que_std
            self.key_proj.weight.data = torch.randn_like(self.key_proj.weight.data) * key_std
            self.val_proj.weight.data = torch.randn_like(self.val_proj.weight.data) * val_std

    def get_lora_parameters(self):
        return [
            self.que_proj.lora_A.default.weight,
            self.que_proj.lora_B.default.weight,
            self.key_proj.lora_A.default.weight,
            self.key_proj.lora_B.default.weight,
            self.val_proj.lora_A.default.weight,
            self.val_proj.lora_B.default.weight
        ]

    def forward(
            self, 
            hidden_states: torch.Tensor,
            memory_states: torch.Tensor,
            num_kv_head: int = 32,
            num_query_head: int = 32,
            head_dim: int = 128
        ):
        query = self.que_proj(hidden_states).unflatten(-1, (num_query_head, head_dim)).transpose(1,2)
        key = self.key_proj(memory_states).unflatten(-1, (num_kv_head, head_dim)).transpose(1,2)
        value = self.val_proj(memory_states).unflatten(-1, (num_kv_head, head_dim)).transpose(1,2)
        return query, key, value
    

class QKVProj(torch.nn.Module):
    def __init__(self, layer, embed_dim: int = 4096):
        super().__init__()
        kwargs = {
            "device": layer.self_attn.q_proj.weight.data.device,
            "dtype": layer.self_attn.q_proj.weight.data.dtype,
        }
        self.que_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False, **kwargs)
        self.key_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False, **kwargs)
        self.val_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False, **kwargs)

        self.que_proj.weight.data = layer.self_attn.q_proj.weight.data.clone()
        self.key_proj.weight.data = layer.self_attn.k_proj.weight.data.clone()
        self.val_proj.weight.data = layer.self_attn.v_proj.weight.data.clone()

    def get_lora_parameters(self):
        return [
            self.que_proj.lora_A.default.weight,
            self.que_proj.lora_B.default.weight,
            self.key_proj.lora_A.default.weight,
            self.key_proj.lora_B.default.weight,
            self.val_proj.lora_A.default.weight,
            self.val_proj.lora_B.default.weight
        ]

    
    def forward(
            self, 
            activation: torch.Tensor,
            num_query_head: int = 32,
            num_kv_head: int = 32,
            head_dim: int = 128):
        query = self.que_proj(activation).unflatten(-1, (num_query_head, head_dim)).transpose(1,2)
        key = self.key_proj(activation).unflatten(-1, (num_kv_head, head_dim)).transpose(1,2)
        value = self.val_proj(activation).unflatten(-1, (num_kv_head, head_dim)).transpose(1,2)
        return query, key, value
    

class OProj(torch.nn.Module):
    def __init__(self, layer, zero_init=False, embed_dim: int = 4096):
        super().__init__()
        kwargs = {
            "device": layer.self_attn.q_proj.weight.data.device,
            "dtype": layer.self_attn.q_proj.weight.data.dtype,
        }
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False, **kwargs)
        self.out_proj.weight.data = layer.self_attn.o_proj.weight.data.clone()
        
        if zero_init:
            self.out_proj.weight.data.fill_(0)

    def forward(self, activation: torch.Tensor):
        if activation.ndim == 4:
            activation = activation.transpose(1,2).flatten(2)
        output = self.out_proj(activation)
        return output


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, layer, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        kwargs = {
            "device": layer.self_attn.q_proj.weight.data.device,
            "dtype": layer.self_attn.q_proj.weight.data.dtype,
        }

        self.weight = torch.nn.Parameter(torch.ones(hidden_size, **kwargs))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)