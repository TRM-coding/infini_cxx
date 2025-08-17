#include "./src/kernels.cu"
#include <vector>
#include <iostream>
#include <cassert>
#include <iomanip>
#include"tester/utils.h"

int main()
{
    // 给定参数
    const int batch_size = 2;
    const int target_seq_len = 3;
    const int src_seq_len = 2;
    const int query_heads = 4;
    const int kv_heads = 2;
    const int head_dim = 5;
    const bool is_causal = false;
    (void)is_causal;


    // 分配
    std::vector<float> h_q = {
        //batch1

        //q_head1
        1, 2, 3, 4, 5,
        1, 3, 2, 4, 5,
           
        //q_head2
        2, 1, 3, 4, 5,
        1, 3, 5, 4, 2,

        //q_head3
        5, 1, 3, 2, 4,
        3, 1, 3, 5, 4,

        //q_head4
        5, 3, 1, 2, 5,
        3, 5, 1, 3, 5,

        //batch 2

        //q_head1
        1, 2, 3, 4, 5,
        1, 3, 2, 4, 5,

        //q_head2
        2, 1, 3, 4, 5,
        1, 3, 5, 4, 2,

        //q_head3
        5, 1, 3, 2, 4,
        3, 1, 3, 5, 4,

        //q_head4
        5, 3, 1, 2, 5,
        3, 5, 1, 3, 5,
    };
    std::vector<float> h_k = {
        //batch1

        //kv_head1
        1, 2, 3, 4, 5,
        2, 3, 1, 4, 5,

        //kv_head2  
        3, 1, 2, 4, 5,
        1, 4, 2, 3, 5,

        //batch2

        //kv_head1
        1, 2, 3, 4, 5,
        2, 3, 1, 4, 5,

        //kv_head2
        3, 1, 2, 4, 5,
        1, 4, 2, 3, 5,
    };
    std::vector<float> h_v = {
        //batch1

        //kv_head1
        2, 1, 4, 3, 5,
        1, 2, 3, 5, 4,

        //kv_head2
        4, 2, 1, 3, 5,
        2, 1, 4, 5, 3,

        //batch2

        //kv_head1
        2, 1, 4, 3, 5,
        1, 2, 3, 5, 4,

        //kv_head2
        4, 2, 1, 3, 5,
        2, 1, 4, 5, 3,
    };
    std::vector<float> h_o(batch_size * query_heads * target_seq_len * head_dim, float(0));

    flashAttention(h_q, h_k, h_v, h_o,
                   batch_size, target_seq_len, src_seq_len,
                   query_heads, kv_heads, head_dim, is_causal);
    
    // 设置输出格式：固定小数点，3位小数，右对齐，宽度10
    std::cout << std::fixed << std::setprecision(3);
    
    for(int i1=0;i1<2;i1++){
        std::cout<<"batch"<<i1<<":"<<std::endl;
        for(int j1=0;j1<4;j1++)
        {
            std::cout<<"  q_head"<<j1<<":"<<std::endl;
            for(int k1=0;k1<target_seq_len;k1++)
            {
                std::cout<<"    ";
                for(int k2=0;k2<head_dim;k2++)
                {
                    auto idx=i1*query_heads*target_seq_len*head_dim+j1*target_seq_len*head_dim+k1*head_dim+k2;
                    std::cout<<std::setw(8)<<h_o[idx];
                    if(k2 < head_dim-1) std::cout<<", ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
    return 0;
}