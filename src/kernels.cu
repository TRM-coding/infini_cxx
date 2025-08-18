#include <vector>
#include <iostream>
#include <fstream>
#include "../tester/utils.h"
#include <cassert>
#include <cfloat>
#include <math_constants.h>
// #include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <iomanip>

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
    // Deprecated: kept for reference; replaced by parallel merge passes launched from host
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

// Each block merges one pair of sorted runs of size `width` from `in` into `out`.
// Runs are contiguous: [start, mid) and [mid, end), where start = blockIdx.x * 2*width
template <typename T>
__global__ void mergePassKernel(const T *in, T *out, size_t n, size_t width)
{
    size_t start = blockIdx.x * (2 * width);
    if (start >= n)
        return;

    size_t mid = min(start + width, n);
    size_t end = min(start + 2 * width, n);

    size_t i = start;
    size_t j = mid;
    size_t k = start;

    // Sequential merge within a block; multiple blocks run in parallel across the array
    while (i < mid && j < end)
    {
        if (in[i] <= in[j])
        {
            out[k++] = in[i++];
        }
        else
        {
            out[k++] = in[j++];
        }
    }
    while (i < mid)
    {
        out[k++] = in[i++];
    }
    while (j < end)
    {
        out[k++] = in[j++];
    }
}

template <typename T>
T kthLargest(const std::vector<T> &h_input, size_t k)
{
    if (h_input.size() < k || k == 0)
    {
        return T(-100);
    }
    // Parallel bottom-up merge sort with per-pass GPU kernels
    const int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    const size_t n = h_input.size();
    const size_t bytes = n * sizeof(T);

    T *d_in = nullptr;
    T *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_input.data(), bytes, cudaMemcpyHostToDevice));

    // Iteratively double run width; each kernel launch merges all pairs of runs of current width
    size_t width = 1;
    // Use 1 thread per block (sequential merge within block), many blocks across array
    dim3 block(1);
    while (width < n)
    {
        size_t numMerges = (n + (2 * width) - 1) / (2 * width);
        dim3 grid((unsigned int)numMerges);
        mergePassKernel<T><<<grid, block>>>(d_in, d_out, n, width);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Ping-pong buffers
        T *tmp = d_in;
        d_in = d_out;
        d_out = tmp;

        width <<= 1;
    }

    // After loop, sorted data is in d_in due to final swap
    T result{};
    size_t idx = n - k; // 0-based index of k-th largest in ascending-sorted array
    CUDA_CHECK(cudaMemcpy(&result, d_in + idx, sizeof(T), cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);

    return result;
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

/**
 * @brief Computes simple attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, target_seq_len, q_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size,src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, target_seq_len, q_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */

// Standard Attention CUDA kernel - supports grouped query attention
template <typename T>
__global__ void flashAttentionKernel(
    const T *q, const T *k, const T *v, T *o,
    size_t batch_size, size_t target_seq_len, size_t src_seq_len,
    size_t query_heads, size_t kv_heads, size_t head_dim, bool is_causal)
{
    size_t batch_idx = blockIdx.y;
    size_t query_head_idx = blockIdx.x;
    size_t target_idx = threadIdx.x;

    if (batch_idx >= batch_size || query_head_idx >= query_heads || target_idx >= target_seq_len)
    {
        return;
    }

    size_t kv_head_idx = ((size_t)query_head_idx * (size_t)kv_heads) / (size_t)query_heads;

    size_t q_batch_offset = (size_t)batch_idx * target_seq_len * query_heads * head_dim;
    size_t k_batch_offset = (size_t)batch_idx * src_seq_len * kv_heads * head_dim;
    size_t v_batch_offset = (size_t)batch_idx * src_seq_len * kv_heads * head_dim;
    size_t o_batch_offset = (size_t)batch_idx * target_seq_len * query_heads * head_dim;

    const T *current_q = q + q_batch_offset + (size_t)target_idx * query_heads * head_dim + (size_t)query_head_idx * head_dim;
    T *current_o = o + o_batch_offset + (size_t)target_idx * query_heads * head_dim + (size_t)query_head_idx * head_dim;

    float scale_factor = 1.0f / sqrtf((float)head_dim);

    // Flash Attention 状态变量
    float m = -CUDART_INF_F;  // 当前最大值
    float l = 0.0f;           // 当前指数和
    
    // 初始化输出为0
    for (int d = 0; d < head_dim; ++d)
    {
        current_o[d] = 0.0f;
    }

    // Flash Attention 主循环 - 逐个处理每个key-value对
    for (int s = 0; s < src_seq_len; ++s)
    {
        // 检查causal masking
        if (is_causal && s > target_idx)
        {
            continue;
        }

        const T *kp = k + k_batch_offset + (size_t)s * kv_heads * head_dim + (size_t)kv_head_idx * head_dim;
        const T *vp = v + v_batch_offset + (size_t)s * kv_heads * head_dim + (size_t)kv_head_idx * head_dim;

        // 计算当前的注意力分数
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d)
        {
            score += (float)current_q[d] * (float)kp[d];
        }
        score *= scale_factor;

        // Flash Attention 更新逻辑
        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float beta = expf(score - m_new);
        
        float l_new = alpha * l + beta;
        
        // 更新输出 O = (l * O + beta * V) / l_new
        for (int d = 0; d < head_dim; ++d)
        {
            current_o[d] = (alpha * l * current_o[d] + beta * (float)vp[d]) / l_new;
        }
        
        // 更新状态
        m = m_new;
        l = l_new;
    }
}

int idx = 0;

template <typename T>
void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k,
                    const std::vector<T> &h_v, std::vector<T> &h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal)
{
    cudaSetDevice(0);
    size_t q_size = h_q.size();
    size_t k_size = h_k.size();
    size_t v_size = h_v.size();
    size_t o_size = h_o.size();
    // std::cout << std::fixed << std::setprecision(12) << "\n";
    // for(auto x:h_q)
    // {
    // std::cout<<x<<" ";
    // }
    // std::cin>>a;

    // std::ofstream f("out/" + std::to_string(idx++) + ".txt");
    // f << std::fixed << std::setprecision(7);
    // f << batch_size << " " << target_seq_len << " " << src_seq_len << " " << kv_heads << " " << query_heads << " " << head_dim << " " << (is_causal ? 1 : 0) << std::endl;
    // for (auto x : h_q)
    // {
    //     f << x << " ";
    // }
    // f << std::endl;
    // for (auto x : h_k)
    // {
    //     f << x << " ";
    // }
    // f << std::endl;
    // for (auto x : h_v)
    // {
    //     f << x << " ";
    // }
    // f << std::endl;
    // f << std::fixed << std::setprecision(12) << "\n";
    // for (auto x : h_q)
    // {
    //     f << x << " ";
    // }
    // f.close();

    float *d_q, *d_k, *d_v, *d_o;

    CUDA_CHECK(cudaMalloc(&d_q, q_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, k_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, v_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o, o_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), k_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), v_size * sizeof(float), cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int threads_per_block = prop.maxThreadsPerBlock;

    dim3 block_size(threads_per_block);
    dim3 grid_size(query_heads, batch_size);

    flashAttentionKernel<<<grid_size, block_size>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim, is_causal);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, o_size * sizeof(float), cudaMemcpyDeviceToHost));


    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int kthLargest<int>(const std::vector<int> &, size_t);
template float kthLargest<float>(const std::vector<float> &, size_t);
template void flashAttention<float>(const std::vector<float> &, const std::vector<float> &,
                                    const std::vector<float> &, std::vector<float> &,
                                    int, int, int, int, int, int, bool);