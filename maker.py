import random
if __name__ == "__main__":

    batch_size=random.randint(1,10)

    target_seq_len=random.randint(1,20)

    src_seq_len=random.randint(1,20)

    kv_heads=random.randint(1,4)

    query_heads=random.randint(1,4)*kv_heads
    # query_heads=kv_heads

    head_dim=random.randint(6,10)

    is_causal=random.randint(0,1)
    
    h_q=[]
    h_k=[]
    h_v=[]

    for _ in range(batch_size):
        for _ in range(query_heads):
            for _ in range(target_seq_len):
                for _ in range(head_dim):
                    h_q.append(random.uniform(0.0, 10.0))

    for _ in range(batch_size):
        for _ in range(kv_heads):
            for _ in range(src_seq_len):
                for _ in range(head_dim):
                    h_k.append(random.uniform(0.0, 10.0))

    for _ in range(batch_size):
        for _ in range(kv_heads):
            for _ in range(src_seq_len):
                for _ in range(head_dim):
                    h_v.append(random.uniform(0.0, 10.0))
    
    # 保存数据到文件
    with open('./compare/input.tx', 'w') as f:
        # 第一行：变量值，空格分隔
        f.write(f"{batch_size} {target_seq_len} {src_seq_len} {kv_heads} {query_heads} {head_dim} {is_causal}\n")
        
       
        h_q_str = ' '.join([f"{x:.7f}" for x in h_q])
        f.write(f"{h_q_str}\n")
        
       
        h_k_str = ' '.join([f"{x:.7f}" for x in h_k])
        f.write(f"{h_k_str}\n")
        
        h_v_str = ' '.join([f"{x:.7f}" for x in h_v])
        f.write(f"{h_v_str}\n")
    
    print(f"数据已保存到 ./compare/input.tx")
    print(f"batch_size={batch_size}, target_seq_len={target_seq_len}, src_seq_len={src_seq_len}")
    print(f"kv_heads={kv_heads}, query_heads={query_heads}, head_dim={head_dim}, is_causal={is_causal}")