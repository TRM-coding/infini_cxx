#include "./src/kernels.cu"
#include <vector>
#include <iostream>
#include <cassert>
#include <cassert>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include"tester/utils.h"
// #define float double


bool loadTestData(const std::string& filename, 
                  std::vector<float>& h_q, std::vector<float>& h_k, std::vector<float>& h_v,
                  int& batch_size, int& target_seq_len, int& src_seq_len,
                  int& kv_heads, int& query_heads, int& head_dim, bool& is_causal) {
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        return false;
    }
    
    std::string line;
    
    // 读取第一行参数
    if (!std::getline(inFile, line)) {
        std::cerr << "Error: Cannot read parameters line" << std::endl;
        return false;
    }
    
    std::istringstream paramStream(line);
    int is_causal_int;
    if (!(paramStream >> batch_size >> target_seq_len >> src_seq_len >> 
          kv_heads >> query_heads >> head_dim >> is_causal_int)) {
        std::cerr << "Error: Cannot parse parameters" << std::endl;
        return false;
    }
    is_causal = (is_causal_int != 0);
    
    // 计算期望的数据长度
    int expected_q_size = batch_size * query_heads * target_seq_len * head_dim;
    int expected_kv_size = batch_size * kv_heads * src_seq_len * head_dim;
    
    // 预分配空间
    h_q.reserve(expected_q_size);
    h_k.reserve(expected_kv_size);
    h_v.reserve(expected_kv_size);
    
    // 读取h_q数据
    if (!std::getline(inFile, line)) {
        std::cerr << "Error: Cannot read h_q data line" << std::endl;
        return false;
    }
    std::istringstream qStream(line);
    float value;
    while (qStream >> value) {
        h_q.push_back(value);
    }
    
    // 读取h_k数据
    if (!std::getline(inFile, line)) {
        std::cerr << "Error: Cannot read h_k data line" << std::endl;
        return false;
    }
    std::istringstream kStream(line);
    while (kStream >> value) {
        h_k.push_back(value);
    }
    
    // 读取h_v数据
    if (!std::getline(inFile, line)) {
        std::cerr << "Error: Cannot read h_v data line" << std::endl;
        return false;
    }
    std::istringstream vStream(line);
    while (vStream >> value) {
        h_v.push_back(value);
    }
    
    inFile.close();
    
    // 验证数据长度
    if (h_q.size() != expected_q_size) {
        std::cerr << "Error: h_q size mismatch. Expected: " << expected_q_size 
                  << ", Got: " << h_q.size() << std::endl;
        return false;
    }
    
    if (h_k.size() != expected_kv_size) {
        std::cerr << "Error: h_k size mismatch. Expected: " << expected_kv_size 
                  << ", Got: " << h_k.size() << std::endl;
        return false;
    }
    
    if (h_v.size() != expected_kv_size) {
        std::cerr << "Error: h_v size mismatch. Expected: " << expected_kv_size 
                  << ", Got: " << h_v.size() << std::endl;
        return false;
    }
    
    return true;
}


int main()
{
    // 声明变量
    int batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim;
    bool is_causal;
    
    // 从文件加载测试数据
    std::vector<float> h_q, h_k, h_v;
    
    if (!loadTestData("./compare/input.tx", h_q, h_k, h_v,
                      batch_size, target_seq_len, src_seq_len,
                      kv_heads, query_heads, head_dim, is_causal)) {
        std::cerr << "Failed to load test data" << std::endl;
        return 1;
    }
    
    std::cout << "Loaded parameters: B=" << batch_size << ", Tq=" << target_seq_len 
              << ", S=" << src_seq_len << ", kvH=" << kv_heads << ", qH=" << query_heads 
              << ", Dh=" << head_dim << ", is_causal=" << is_causal << std::endl;

    std::vector<float> h_o(batch_size * query_heads * target_seq_len * head_dim, float(0));

    flashAttention(h_q, h_k, h_v, h_o,
                   batch_size, target_seq_len, src_seq_len,
                   query_heads, kv_heads, head_dim, is_causal);
    
    // 将结果输出到文件
    std::ofstream outFile("./compare/out.test");
    if (!outFile.is_open()) {
        std::cerr << "Error: Cannot create out.test" << std::endl;
        return 1;
    }
    
    outFile << std::fixed << std::setprecision(8);
    
    // 只输出数值，保留8位小数，空格分隔
    for(size_t i = 0; i < h_o.size(); i++) {
        if (i > 0) outFile << " ";
        outFile << h_o[i];
    }
    
    outFile.close();
    
    std::cout << "Output generated successfully: out.test" << std::endl;
    
    return 0;
}