#include "./src/kernels.cu"
#include <vector>
#include <iostream>
#include <cassert>
#include <cassert>
#include <iomanip>
#include"tester/utils.h"

int main()
{
    // 给定参数
    const int batch_size = 1;
    const int target_seq_len = 8;
    const int src_seq_len = 8;
    const int query_heads = 8;
    const int kv_heads = 4;
    const int head_dim = 4;
    const bool is_causal = true;
    (void)is_causal;


    // 分配随机化的输入数据（包含小数）
    // Query tensor: [batch_size=1, query_heads=8, target_seq_len=8, head_dim=4]
    std::vector<float> h_q = {
        // batch 0, q_head 0
        0.5f, 1.2f, -0.8f, 2.1f,   // seq 0
        1.3f, -0.4f, 2.0f, 0.9f,   // seq 1
        -0.6f, 2.3f, 1.1f, -0.9f,  // seq 2
        0.8f, 1.7f, -1.2f, 0.3f,   // seq 3
        2.1f, -0.5f, 0.9f, 1.6f,   // seq 4
        -0.2f, 1.4f, 0.6f, -2.0f,  // seq 5
        1.8f, 0.1f, -1.5f, 0.7f,   // seq 6
        -0.7f, 1.9f, 0.5f, 1.0f,   // seq 7
        
        // batch 0, q_head 1
        -1.4f, 0.8f, 2.5f, -0.1f,  // seq 0
        0.3f, -1.8f, 1.2f, 0.9f,   // seq 1
        1.1f, -0.9f, 0.4f, 2.0f,   // seq 2
        -0.5f, 1.3f, -2.1f, 0.8f,  // seq 3
        0.2f, 1.9f, -1.0f, -0.4f,  // seq 4
        1.5f, -0.8f, 0.7f, -1.6f,  // seq 5
        -2.2f, 0.6f, 1.8f, -0.2f,  // seq 6
        0.9f, -1.5f, -0.7f, 1.7f,  // seq 7
        
        // batch 0, q_head 2
        2.3f, -1.1f, 0.4f, 1.8f,   // seq 0
        -0.6f, 1.9f, -1.4f, 0.2f,  // seq 1
        1.6f, 0.7f, -2.0f, 1.3f,   // seq 2
        -1.8f, 0.5f, 2.1f, -0.9f,  // seq 3
        0.1f, -1.7f, 1.4f, 2.2f,   // seq 4
        1.0f, -0.3f, -1.9f, 0.8f,  // seq 5
        -2.1f, 1.6f, 0.3f, -1.2f,  // seq 6
        0.7f, 2.4f, -0.8f, -0.4f,  // seq 7
        
        // batch 0, q_head 3
        1.2f, -2.3f, 0.9f, 1.7f,   // seq 0
        -0.1f, 1.5f, -1.8f, 0.6f,  // seq 1
        2.0f, -0.7f, 1.1f, -1.4f,  // seq 2
        0.4f, 1.9f, -0.2f, 2.1f,   // seq 3
        -1.6f, 0.8f, 1.3f, -0.5f,  // seq 4
        2.2f, -1.0f, -0.9f, 1.8f,  // seq 5
        0.3f, -2.4f, 1.7f, 0.0f,   // seq 6
        -1.3f, 0.9f, 2.5f, -1.1f,  // seq 7
        
        // batch 0, q_head 4
        -0.8f, 1.4f, 2.0f, -1.7f,  // seq 0
        1.1f, -0.6f, 0.5f, 2.3f,   // seq 1
        -2.1f, 0.2f, 1.8f, -0.4f,  // seq 2
        0.9f, 1.6f, -1.2f, 0.7f,   // seq 3
        -1.9f, -0.1f, 2.4f, 1.0f,  // seq 4
        1.5f, -2.0f, 0.8f, -1.5f,  // seq 5
        0.6f, 1.3f, -0.9f, 2.2f,   // seq 6
        -1.8f, 0.4f, 1.9f, -0.3f,  // seq 7
        
        // batch 0, q_head 5
        2.1f, -1.0f, -0.7f, 1.6f,  // seq 0
        0.3f, 1.8f, -2.2f, 0.5f,   // seq 1
        -1.4f, 0.9f, 2.0f, -0.8f,  // seq 2
        1.7f, -0.2f, -1.1f, 2.4f,  // seq 3
        0.1f, -1.9f, 1.3f, 0.6f,   // seq 4
        -2.3f, 1.2f, 0.4f, -1.6f,  // seq 5
        1.5f, 0.8f, -0.5f, 1.9f,   // seq 6
        -0.9f, 2.1f, 1.0f, -1.7f,  // seq 7
        
        // batch 0, q_head 6
        0.7f, -2.0f, 1.4f, 0.2f,   // seq 0
        -1.3f, 1.8f, -0.6f, 2.2f,  // seq 1
        1.1f, -0.4f, 2.5f, -1.8f,  // seq 2
        -0.9f, 1.6f, 0.3f, -2.1f,  // seq 3
        2.3f, -1.2f, 0.8f, 1.4f,   // seq 4
        -0.1f, -1.9f, 1.7f, 0.5f,  // seq 5
        1.0f, 2.4f, -1.5f, -0.7f,  // seq 6
        -2.2f, 0.6f, 1.9f, 1.3f,   // seq 7
        
        // batch 0, q_head 7
        -1.6f, 0.4f, 2.1f, -0.9f,  // seq 0
        1.8f, -2.3f, 0.7f, 1.2f,   // seq 1
        -0.5f, 1.9f, -1.4f, 0.1f,  // seq 2
        2.0f, -0.8f, 1.5f, -2.4f,  // seq 3
        0.6f, 1.7f, -0.3f, 1.1f,   // seq 4
        -1.8f, -0.2f, 2.2f, 0.9f,  // seq 5
        1.4f, -1.1f, 0.5f, -1.9f,  // seq 6
        -0.7f, 2.5f, 1.6f, 0.3f,   // seq 7
    };
    
    // Key tensor: [batch_size=1, kv_heads=4, src_seq_len=8, head_dim=4]
    std::vector<float> h_k = {
        // batch 0, kv_head 0
        1.4f, -0.3f, 0.8f, -1.9f,  // seq 0
        -0.7f, 2.0f, 1.2f, 0.5f,   // seq 1
        0.9f, -1.1f, -0.6f, 1.8f,  // seq 2
        -1.3f, 0.4f, 1.7f, -0.8f,  // seq 3
        2.2f, -0.1f, -1.6f, 0.7f,  // seq 4
        0.3f, 1.9f, -0.5f, -1.2f,  // seq 5
        -0.9f, 1.5f, 0.1f, 2.3f,   // seq 6
        1.1f, -2.1f, 0.8f, -0.4f,  // seq 7
        
        // batch 0, kv_head 1
        -1.0f, 0.7f, 2.4f, -0.2f,  // seq 0
        0.6f, -1.7f, 1.3f, 0.9f,   // seq 1
        -2.2f, 0.5f, -1.4f, 1.8f,  // seq 2
        1.2f, -0.8f, 0.4f, -1.9f,  // seq 3
        2.0f, 1.6f, -0.3f, 0.1f,   // seq 4
        -1.5f, 2.1f, 0.8f, -1.1f,  // seq 5
        0.9f, -0.6f, 1.7f, 2.3f,   // seq 6
        -2.4f, 1.0f, -0.7f, 1.4f,  // seq 7
        
        // batch 0, kv_head 2
        1.8f, -1.2f, 0.5f, 2.1f,   // seq 0
        -0.4f, 1.6f, -2.0f, 0.3f,  // seq 1
        2.2f, 0.8f, -1.7f, -0.9f,  // seq 2
        -1.3f, 2.5f, 1.1f, 0.6f,   // seq 3
        0.2f, -1.8f, 1.9f, -1.4f,  // seq 4
        1.5f, 0.7f, -0.1f, 2.0f,   // seq 5
        -2.3f, 1.4f, 0.9f, -0.5f,  // seq 6
        1.7f, -0.8f, 2.4f, 1.2f,   // seq 7
        
        // batch 0, kv_head 3
        -0.6f, 2.1f, 1.0f, -1.5f,  // seq 0
        1.9f, -1.1f, 0.4f, 2.2f,   // seq 1
        -2.0f, 0.8f, 1.6f, -0.3f,  // seq 2
        1.3f, 2.4f, -1.8f, 0.7f,   // seq 3
        -0.9f, -0.2f, 1.7f, 1.1f,  // seq 4
        2.3f, 1.5f, -1.4f, -0.8f,  // seq 5
        0.1f, -1.9f, 2.5f, 0.5f,   // seq 6
        -1.6f, 1.8f, 0.9f, -2.1f,  // seq 7
    };
    
    // Value tensor: [batch_size=1, kv_heads=4, src_seq_len=8, head_dim=4]
    std::vector<float> h_v = {
        // batch 0, kv_head 0
        2.0f, -1.2f, 0.6f, 1.8f,   // seq 0
        -0.9f, 1.5f, -2.1f, 0.3f,  // seq 1
        0.8f, -0.1f, 1.9f, -1.6f,  // seq 2
        -1.4f, 0.7f, 2.2f, -0.5f,  // seq 3
        1.1f, -2.0f, 0.4f, 1.6f,   // seq 4
        -0.3f, 1.3f, -1.1f, 2.4f,  // seq 5
        1.7f, -0.6f, -2.3f, 0.1f,  // seq 6
        -1.9f, 0.8f, 1.0f, -1.5f,  // seq 7
        
        // batch 0, kv_head 1
        0.5f, 1.8f, -0.2f, -1.3f,  // seq 0
        -0.4f, 2.5f, 1.2f, -1.0f,  // seq 1
        1.9f, -0.8f, -1.7f, 0.3f,  // seq 2
        -1.6f, 0.9f, 2.0f, 1.1f,   // seq 3
        -0.5f, -2.2f, 1.4f, 0.7f,  // seq 4
        2.1f, 1.6f, -1.9f, -0.1f,  // seq 5
        0.8f, -1.4f, 2.3f, 1.8f,   // seq 6
        -2.4f, 0.2f, -0.9f, 1.7f,  // seq 7
        
        // batch 0, kv_head 2
        1.3f, -2.1f, 0.9f, 1.5f,   // seq 0
        -0.7f, 1.8f, -1.2f, 0.4f,  // seq 1
        2.2f, 0.1f, -1.9f, -0.6f,  // seq 2
        -1.1f, 2.0f, 1.7f, 0.8f,   // seq 3
        0.3f, -1.5f, 2.4f, -1.8f,  // seq 4
        1.6f, 0.9f, -0.4f, 2.1f,   // seq 5
        -2.0f, 1.2f, 0.5f, -1.7f,  // seq 6
        1.4f, -0.3f, 2.3f, 0.7f,   // seq 7
        
        // batch 0, kv_head 3
        -0.8f, 1.9f, 1.1f, -2.2f,  // seq 0
        2.5f, -1.3f, 0.6f, 1.7f,   // seq 1
        -1.8f, 0.4f, 2.1f, -0.9f,  // seq 2
        1.0f, 2.3f, -1.6f, 0.2f,   // seq 3
        -0.5f, -1.1f, 1.8f, 1.4f,  // seq 4
        2.0f, 1.2f, -2.4f, -0.7f,  // seq 5
        0.9f, -1.7f, 2.2f, 0.3f,   // seq 6
        -1.4f, 1.6f, 0.8f, -2.0f,  // seq 7
    };
    std::vector<float> h_o(batch_size * query_heads * target_seq_len * head_dim, float(0));

    flashAttention(h_q, h_k, h_v, h_o,
                   batch_size, target_seq_len, src_seq_len,
                   query_heads, kv_heads, head_dim, is_causal);
    
    std::cout << std::fixed << std::setprecision(5);
    
    for(int i1=0; i1<batch_size; i1++){
        std::cout<<"batch"<<i1<<":"<<std::endl;
        for(int j1=0; j1<query_heads; j1++)
        {
            std::cout<<"  q_head"<<j1<<":"<<std::endl;
            for(int k1=0; k1<target_seq_len; k1++)
            {
                std::cout<<"    ";
                for(int k2=0; k2<head_dim; k2++)
                {
                    auto idx=i1*query_heads*target_seq_len*head_dim+j1*target_seq_len*head_dim+k1*head_dim+k2;
                    std::cout<<std::setw(8)<<h_o[idx]<<std::setprecision(5);;
                    if(k2 < head_dim-1) std::cout<<", ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }

    for(auto x:h_o)
    {
        std::cout<<x<<", ";
    }
    return 0;
}
