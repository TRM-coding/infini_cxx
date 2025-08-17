import torch
import math

#softmax((QK^T)/sqrt(head_dim))V
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    # query = query.to(torch.float64)
    # key = key.to(torch.float64)
    # value = value.to(torch.float64)
    # attn_bias = attn_bias.to(torch.float64)

    print("max_query:",query.max().item()," max_key:",key.max().item()," max_value:",value.max().item())
    print("min_query:",query.min().item()," min_key:",key.min().item()," min_value:",value.min().item()) 
    print(scale_factor)  
    print(query.shape)
    attn_weight = (query.to(torch.float64) @ key.transpose(-2, -1).to(torch.float64)).to(torch.float64) * scale_factor + attn_bias
    attn_weight = torch.softmax(attn_weight.to(torch.float64), dim=-1).to(torch.float32)
    # attn_weight = torch.softmax(query @ (key.transpose(-2, -1) * scale_factor) + attn_bias, dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    # return attn_weight @ v
    # alue
    return attn_weight @ value


import torch
import torch.nn.functional as F
import sys
import os



# 固定打印格式
torch.set_printoptions(precision=9, sci_mode=False)

def load_test_data(filename="./compare/input.tx"):
    """
    从文件中加载测试数据
    文件格式：
    第一行：batch_size target_seq_len src_seq_len kv_heads query_heads head_dim is_causal
    第二行：h_q数据（空格分隔的浮点数）
    第三行：h_k数据（空格分隔的浮点数）
    第四行：h_v数据（空格分隔的浮点数）
    """
    if not os.path.exists(filename):
        print(f"Error: {filename} not found!")
        sys.exit(1)
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 解析第一行参数
    params = list(map(int, lines[0].strip().split()))
    batch_size, target_seq_len, src_seq_len, kv_heads, query_heads, head_dim, is_causal = params
    
    # 解析h_q数据
    h_q = list(map(float, lines[1].strip().split()))
    
    # 解析h_k数据
    h_k = list(map(float, lines[2].strip().split()))
    
    # 解析h_v数据
    h_v = list(map(float, lines[3].strip().split()))
    
    return h_q, h_k, h_v, batch_size, target_seq_len, src_seq_len, kv_heads, query_heads, head_dim, is_causal

# 从文件加载数据
h_q, h_k, h_v, B, Tq, S, kvH, qH, Dh, is_causal = load_test_data()

# 验证数据长度
assert len(h_q) == B*qH*Tq*Dh, f"h_q length mismatch: expected {B*qH*Tq*Dh}, got {len(h_q)}"
assert len(h_k) == B*kvH*S*Dh, f"h_k length mismatch: expected {B*kvH*S*Dh}, got {len(h_k)}"
assert len(h_v) == B*kvH*S*Dh, f"h_v length mismatch: expected {B*kvH*S*Dh}, got {len(h_v)}"

# 构造张量
q = torch.tensor(h_q, dtype=torch.float32).view(B, qH, Tq, Dh)
k = torch.tensor(h_k, dtype=torch.float32).view(B, kvH, S, Dh)
v = torch.tensor(h_v, dtype=torch.float32).view(B, kvH, S, Dh)

# GQA：将 K/V 沿 head 维复制到与 qH 对齐
group_size = qH // kvH
assert qH % kvH == 0, f"qH ({qH}) must be divisible by kvH ({kvH})"
# k = k.repeat_interleave(group_size, dim=1)  # [B, qH, S, Dh]
# v = v.repeat_interleave(group_size, dim=1)  # [B, qH, S, Dh]
from torch.backends.cuda import sdp_kernel
# 计算 SDPA（等价于缩放点积注意力），默认 scale=1/sqrt(Dh)

o = scaled_dot_product_attention(q, k, v, is_causal=bool(is_causal), dropout_p=0.0, enable_gqa=True)  # [B, qH, Tq, Dh]

# 将结果写入文件，只输出数值，保留6位小数，空格分隔
with open("compare/out.test", "w") as f:
    # 获取扁平化数据
    o_flat = o.contiguous().view(-1).tolist()
    
    # 写入数据，保留6位小数，空格分隔
    output_str = ' '.join([f"{x:.6f}" for x in o_flat])
    f.write(output_str)

print("Answer generated successfully: answer.test")
print(f"Loaded parameters: B={B}, Tq={Tq}, S={S}, kvH={kvH}, qH={qH}, Dh={Dh}, is_causal={is_causal}")