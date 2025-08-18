#include <algorithm>
#include <cassert>
#include <cmath>
#include <cfloat>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>

#include "../tester/utils.h"

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

void softmax(const float *input, float *output, int length)
{
    float max_val = *std::max_element(input, input + length);
    float sum = 0.0f;
    for (int i = 0; i < length; ++i)
    {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < length; ++i)
    {
        output[i] /= sum;
    }
}
/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k,
                    const std::vector<T> &h_v, std::vector<T> &h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal)
{
    // h_o.resize(batch_size * target_seq_len * query_heads * head_dim);
    const T scale = static_cast<T>(1.0 / std::sqrt(static_cast<float>(head_dim)));
    auto inf = std::numeric_limits<T>::infinity();

    auto idxQ = [&](int b, int tq, int h, int d)
    {
        return ((size_t)b * target_seq_len * query_heads * head_dim) +
               ((size_t)tq * query_heads + h) * head_dim + d;
    };
    auto idxK = [&](int b, int sk, int kvh, int d)
    {
        return ((size_t)b * src_seq_len * kv_heads * head_dim) +
               ((size_t)sk * kv_heads + kvh) * head_dim + d;
    };

    auto idxKV = [&](int b,int sk,int kvh,int d){
        return ((size_t)b*src_seq_len*kv_heads*head_dim) +
               ((size_t)sk*kv_heads + kvh)*head_dim + d;
    };

    auto idxV = idxK; // 同布局
    auto idxO = idxQ; // 同布局

    std::vector<T> scores(src_seq_len), weights(src_seq_len);
    std::vector<T> acc(head_dim);

    for (int b = 0; b < batch_size; ++b)
    {
        for (int h = 0; h < query_heads; ++h)
        {
            const int kvh = (h * kv_heads) / query_heads;
            for (int tq = 0; tq < target_seq_len; ++tq)
            {
                const T *q = &h_q[idxQ(b, tq, h, 0)];
                // 在线 softmax 统计
                T m = -inf;                              // running max
                T s = T(0);                              // running sum of exp
                std::fill(acc.begin(), acc.end(), T(0)); // running sum exp(score - m) * V

                for (int sk = 0; sk < src_seq_len; ++sk)
                {
                    if (is_causal && sk > tq)
                        continue;

                    const T *k = &h_k[idxKV(b, sk, kvh, 0)];
                    const T *v = &h_v[idxKV(b, sk, kvh, 0)];

                    T dot = 0;
                    for (int d = 0; d < head_dim; ++d)
                        dot += q[d] * k[d];
                    T score = dot * scale;

                    // 更新 (m,s,acc)
                    T m_new = std::max(m, score);
                    T alpha = (m == -inf) ? T(0) : std::exp(m - m_new);
                    T e = std::exp(score - m_new);

                    s = s * alpha + e;
                    for (int d = 0; d < head_dim; ++d)
                        acc[d] = acc[d] * alpha + e * v[d];

                    m = m_new;
                }

                T inv_s = (s > T(0)) ? T(1) / s : T(0);
                for (int d = 0; d < head_dim; ++d)
                    h_o[idxQ(b, tq, h, d)] = acc[d] * inv_s;
            }
        }
    }
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