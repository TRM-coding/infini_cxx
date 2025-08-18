#include <vector>

#include "../tester/utils.h"
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <cassert>

// CUDA headers
#include <cuda_runtime.h>
#include <math_constants.h>
#include <iostream> 

/**
 * @brief Find the k-th largest element in a vector using CUDA.
 *
 * @tparam T Type of elements in the input vector (should support `int` and `float`).
 * @param h_input Host-side input vector.
 * @param k 1-based index of the element to find (e.g., `k=1` returns the largest element).
 * @return T The k-th largest element in `h_input`.

 * @note Must use CUDA kernels for all compute-intensive steps; no significant CPU allowed.
 * @note Library functions that can directly complete a significant part of the work are NOT allowed.
 * @note For invalid cases, return T(-100).
 * @note Handles device memory management (allocate/copy/free) internally. Errors should be thrown.
 */
template <typename T>
__device__ void merge(T *data, size_t left, size_t mid, size_t right, T *temp)
{
    size_t i = left, j = mid, k = 0;

    while (i < mid && j < right)
    {
        if (data[i] <= data[j])
        {
            temp[k++] = data[i++];
        }
        else
        {
            temp[k++] = data[j++];
        }
    }

    while (i < mid)
    {
        temp[k++] = data[i++];
    }

    while (j < right)
    {
        temp[k++] = data[j++];
    }

    for (size_t i = 0; i < k; i++)
    {
        data[left + i] = temp[i];
    }
}

template <typename T>
__device__ void iterative_merge_sort(T *data, size_t n, T *temp)
{
    for (size_t curr_size = 1; curr_size < n; curr_size *= 2)
    {
        for (size_t left_start = 0; left_start < n - 1; left_start += 2 * curr_size)
        {
            size_t mid = fminf(left_start + curr_size - 1, n - 1);
            size_t right_end = fminf(left_start + 2 * curr_size - 1, n - 1);

            if (mid < right_end)
            {
                merge(data, left_start, mid + 1, right_end + 1, temp);
            }
        }
    }
}
template <typename T>
__global__ void KernelKthLargest(T *data, T *ans, size_t length, size_t k, T *temp)
{
    iterative_merge_sort(data, length, temp);
    *ans = data[length - k];
}

template <typename T>
T kthLargest(const std::vector<T> &h_input, size_t k)
{
    if (h_input.size() < k || k == 0)
    {
        return T(-100);
    }
    // TODO: Implement the kthLargest function
    const int device = 0;
    cudaSetDevice(device);
    T *data = nullptr;
    T *ans = new T;
    T *cudaans = nullptr;
    T *temp = nullptr;
    const size_t DATA_SIZE = h_input.size();
    const size_t BYTE_SIZE = DATA_SIZE * sizeof(T);
    CUDA_CHECK(cudaMalloc(&data, BYTE_SIZE));
    CUDA_CHECK(cudaMalloc(&cudaans, sizeof(T)));
    CUDA_CHECK(cudaMalloc(&temp, BYTE_SIZE));
    CUDA_CHECK(cudaMemcpy(data, h_input.data(), BYTE_SIZE, cudaMemcpyHostToDevice));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    KernelKthLargest<<<1, 1>>>(data, cudaans, size_t(h_input.size()), k, temp);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(ans, cudaans, sizeof(T), cudaMemcpyDeviceToHost));
    return *ans;
}

template <typename T>
__device__ void softmax(T *vec, int len)
{
    float sum_exp = 0.0f;
    float max_score = -CUDART_INF_F;
    for (int s = 0; s < len; ++s)
    {
        max_score = fmaxf(max_score, vec[s]);
    }

    for (size_t s = 0; s < len; ++s)
    {
        if (vec[s] == -CUDART_INF_F)
        {
            vec[s] = 0;
            continue;
        }
        vec[s] = expf(vec[s] - max_score);
        sum_exp += vec[s];
    }

    float scal=1.0/(sum_exp+1e-8);

    for (int s = 0; s < len; ++s)
    {
        vec[s] *=scal;
    }
}


// 设备端索引下标函数，与 CPU 版 idxQ/idxKV 一致
__device__ __forceinline__ size_t idxQ_dev(int b, int tq, int h, int d,
                         int target_seq_len, int query_heads, int head_dim) {
    return ((size_t)b * target_seq_len * query_heads * head_dim) +
        ((size_t)tq * query_heads + h) * head_dim + d;
}

__device__ __forceinline__ size_t idxKV_dev(int b, int sk, int kvh, int d,
                          int src_seq_len, int kv_heads, int head_dim) {
    return ((size_t)b * src_seq_len * kv_heads * head_dim) +
        ((size_t)sk * kv_heads + kvh) * head_dim + d;
}
template <typename T>
__global__ void flashAttentionKernel(const T* d_q, const T* d_k, const T* d_v, T* d_o,
                                     int batch_size, int target_seq_len, int src_seq_len,
                                     int query_heads, int kv_heads, int head_dim, bool is_causal) {
    if (threadIdx.x == 0 && blockIdx.x == 0) { // 只用一个线程执行
        extern __shared__ unsigned char smem[];
        T* acc = reinterpret_cast<T*>(smem); // 大小: head_dim

        const T scale = static_cast<T>(1.0f / sqrtf(static_cast<float>(head_dim)));

        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < query_heads; ++h) {
                const int kvh = (h * kv_heads) / query_heads;

                for (int tq = 0; tq < target_seq_len; ++tq) {
                    // q 指针
                    size_t q_base = idxQ_dev(b, tq, h, 0, target_seq_len, query_heads, head_dim);
                    const T* q = d_q + q_base;

                    // 在线 softmax 状态
                    T m = static_cast<T>(-CUDART_INF_F);
                    T s = static_cast<T>(0);

                    // 清零累加器
                    for (int d = 0; d < head_dim; ++d) acc[d] = static_cast<T>(0);

                    for (int sk = 0; sk < src_seq_len; ++sk) {
                        if (is_causal && sk > tq) continue;

                        size_t kv_base = idxKV_dev(b, sk, kvh, 0, src_seq_len, kv_heads, head_dim);
                        const T* k = d_k + kv_base;
                        const T* v = d_v + kv_base;

                        // q·k
                        T dot = static_cast<T>(0);
                        for (int d = 0; d < head_dim; ++d) {
                            dot += q[d] * k[d];
                        }
                        T score = dot * scale;

                        // 更新 (m, s, acc)
                        T m_new = score > m ? score : m;
                        // 当 m 为 -inf 时，alpha 应为 0
                        T alpha = (m == static_cast<T>(-CUDART_INF_F)) ? static_cast<T>(0)
                                                                         : static_cast<T>(expf(static_cast<float>(m - m_new)));
                        T e = static_cast<T>(expf(static_cast<float>(score - m_new)));

                        s = s * alpha + e;
                        for (int d = 0; d < head_dim; ++d) {
                            acc[d] = acc[d] * alpha + e * v[d];
                        }
                        m = m_new;
                    }

                    // 写出结果
                    T inv_s = (s > static_cast<T>(0)) ? static_cast<T>(1) / s : static_cast<T>(0);
                    for (int d = 0; d < head_dim; ++d) {
                        d_o[q_base + d] = acc[d] * inv_s; // 与 q 同布局
                    }
                }
            }
        }
    }
}

// 简单 softmax 实现
void softmax(const float* input, float* output, int length) {
    float max_val = *std::max_element(input, input + length);
    float sum = 0.0f;
    for (int i = 0; i < length; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < length; ++i) {
        output[i] /= sum;
    }
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    if (h_q.size() != static_cast<size_t>(batch_size * target_seq_len * query_heads * head_dim)) {
        throw std::invalid_argument("Invalid q tensor size");
    }
    if (h_k.size() != static_cast<size_t>(batch_size * src_seq_len * kv_heads * head_dim)) {
        throw std::invalid_argument("Invalid k tensor size");
    }
    if (h_v.size() != static_cast<size_t>(batch_size * src_seq_len * kv_heads * head_dim)) {
        throw std::invalid_argument("Invalid v tensor size");
    }

    if (head_dim <= 0) {
        throw std::invalid_argument("head_dim must be > 0");
    }

    h_o.resize(batch_size * target_seq_len * query_heads * head_dim);

    // 分配设备内存
    T* d_q;
    T* d_k;
    T* d_v;
    T* d_o;
    
    size_t q_size = h_q.size() * sizeof(T);
    size_t k_size = h_k.size() * sizeof(T);
    size_t v_size = h_v.size() * sizeof(T);
    size_t o_size = h_o.size() * sizeof(T);
    
    cudaError_t err;
    err = cudaMalloc(&d_q, q_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for q");
    }
    
    err = cudaMalloc(&d_k, k_size);
    if (err != cudaSuccess) {
        cudaFree(d_q);
        throw std::runtime_error("Failed to allocate device memory for k");
    }
    
    err = cudaMalloc(&d_v, v_size);
    if (err != cudaSuccess) {
        cudaFree(d_q);
        cudaFree(d_k);
        throw std::runtime_error("Failed to allocate device memory for v");
    }
    
    err = cudaMalloc(&d_o, o_size);
    if (err != cudaSuccess) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        throw std::runtime_error("Failed to allocate device memory for o");
    }
    
    // 复制数据到设备
    err = cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        throw std::runtime_error("Failed to copy q to device");
    }
    
    err = cudaMemcpy(d_k, h_k.data(), k_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        throw std::runtime_error("Failed to copy k to device");
    }
    
    err = cudaMemcpy(d_v, h_v.data(), v_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        throw std::runtime_error("Failed to copy v to device");
    }
    
    // 启动kernel，动态共享内存用于 acc[head_dim]
    size_t shm_bytes = static_cast<size_t>(head_dim) * sizeof(T);
    flashAttentionKernel<T><<<1, 1, shm_bytes>>>(d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len,
                                                 query_heads, kv_heads, head_dim, is_causal);
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        throw std::runtime_error("Kernel execution failed");
    }
    
    // 复制结果回主机
    err = cudaMemcpy(h_o.data(), d_o, o_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        throw std::runtime_error("Failed to copy result to host");
    }
    
    // 释放设备内存
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}
// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int kthLargest<int>(const std::vector<int>&, size_t);
template float kthLargest<float>(const std::vector<float>&, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);