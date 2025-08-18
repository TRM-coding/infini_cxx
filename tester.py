import torch
import torch.nn.functional as F
import sys
import os

#softmax((QK^T)/sqrt(head_dim))V

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
# q = torch.tensor(h_q, dtype=torch.float32).view(B, Tq, qH, Dh).transpose(-2,-3)
# k = torch.tensor(h_k, dtype=torch.float32).view(B, S, kvH, Dh).transpose(-2,-3)
# v = torch.tensor(h_v, dtype=torch.float32).view(B, S, kvH, Dh).transpose(-2,-3)

q = torch.tensor(h_q, dtype=torch.float32).view(B, Tq, qH, Dh).transpose(-2,-3)
k = torch.tensor(h_k, dtype=torch.float32).view(B, S, kvH, Dh).transpose(-2,-3)
v = torch.tensor(h_v, dtype=torch.float32).view(B, S, kvH, Dh).transpose(-2,-3)

# GQA：将 K/V 沿 head 维复制到与 qH 对齐
group_size = qH // kvH
assert qH % kvH == 0, f"qH ({qH}) must be divisible by kvH ({kvH})"
# k = k.repeat_interleave(group_size, dim=1)  # [B, qH, S, Dh]
# v = v.repeat_interleave(group_size, dim=1)  # [B, qH, S, Dh]
# from torch.backends.cuda import sdp_kernel
# 计算 SDPA（等价于缩放点积注意力），默认 scale=1/sqrt(Dh)

o = F.scaled_dot_product_attention(q, k, v, is_causal=bool(is_causal), dropout_p=0.0, enable_gqa=True)  # [B, qH, Tq, Dh]

o=o.transpose(-2,-3)

# 将结果写入文件，只输出数值，保留6位小数，空格分隔
with open("compare/answer.test", "w") as f:
    # 获取扁平化数据
    o_flat = o.contiguous().view(-1).tolist()
    
    # 写入数据，保留6位小数，空格分隔
    output_str = ' '.join([f"{x:.9f}" for x in o_flat])
    f.write(output_str)

print("Answer generated successfully: answer.test")
print(f"Loaded parameters: B={B}, Tq={Tq}, S={S}, kvH={kvH}, qH={qH}, Dh={Dh}, is_causal={is_causal}")