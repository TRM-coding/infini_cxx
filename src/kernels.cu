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
// #define double long double
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
// template <typename T>
// __host__ __device__ inline void merge_sort(T *data, size_t loc_a, size_t loc_b,T* temp)
// {
//   if (loc_b - loc_a <= 1)
//   {
//     return;
//   }
//   auto mid = (loc_a + loc_b) / 2;
//   merge_sort(data, loc_a, mid,temp);
//   merge_sort(data, mid, loc_b,temp);
//   size_t left = loc_a, right = mid;
//   // T *temp = new T[loc_b-loc_a];
//   size_t cnt = 0;
//   while (left < mid && right < loc_b)
//   {
//     if (data[left] < data[right])
//     {
//       temp[cnt++] = data[left++];
//     }
//     else
//     {
//       temp[cnt++] = data[right++];
//     }
//   }
//   while (left < mid)
//   {
//     temp[cnt++] = data[left++];
//   }
//   while (right < loc_b)
//   {
//     temp[cnt++] = data[right++];
//   }
//   for (size_t i = 0; i < cnt; i++) {
//         data[loc_a + i] = temp[i];
//   }

//   return;
// }

// template <typename T>
// __global__ void KernelKthLargest(T *data, T *ans, size_t lenth, size_t k,T* temp)
// {

//   merge_sort(data, 0, lenth,temp);
//   *ans = data[lenth - k];
//   return;
// }

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

template<typename T>
__device__ void softmax(T* vec,int len)
{
    double sum_exp = 0.0f;
    double max_score = -CUDART_INF_F;
    for (int s = 0; s < len; ++s)
    {
        max_score = fmax(max_score, vec[s]);
    }

    for (int s = 0; s < len; ++s)
    {
        if (vec[s] == -CUDART_INF_F)
        {
            vec[s] = 0;
            continue;
        }
        vec[s] = exp(vec[s] - max_score);
        sum_exp += vec[s];
    }

    for (int s = 0; s < len; ++s)
    {
        vec[s] /= sum_exp;
    }
}


/**
 * @brief Computes simple attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, query_heads, target_seq_len, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, kv_heads, src_seq_len, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, kv_heads, src_seq_len, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, query_heads, target_seq_len, head_dim]
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
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim, bool is_causal)
{
    int batch_idx = blockIdx.z;
    int query_head_idx = blockIdx.y;
    // int target_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int target_idx = threadIdx.x;

    if (batch_idx >= batch_size || query_head_idx >= query_heads || target_idx >= target_seq_len)
    {
        return;
    }

    int kv_head_idx;

    kv_head_idx = (query_head_idx * kv_heads) / query_heads;

    int q_batch_offset = batch_idx * query_heads * target_seq_len * head_dim;
    int k_batch_offset = batch_idx * kv_heads * src_seq_len * head_dim;
    int v_batch_offset = batch_idx * kv_heads * src_seq_len * head_dim;
    int o_batch_offset = batch_idx * query_heads * target_seq_len * head_dim;

    int q_head_offset = query_head_idx * target_seq_len * head_dim;
    int kv_head_offset = kv_head_idx * src_seq_len * head_dim;
    int o_head_offset = query_head_idx * target_seq_len * head_dim;

    const T *current_q = q + q_batch_offset + q_head_offset + target_idx * head_dim;
    const T *current_k_base = k + k_batch_offset + kv_head_offset;
    const T *current_v_base = v + v_batch_offset + kv_head_offset;
    T *current_o = o + o_batch_offset + o_head_offset + target_idx * head_dim;

    double scale_factor = 1.0 / sqrt((double)head_dim);

    double attn_scores[2048];

    for (int s = 0; s < src_seq_len; ++s)
    {
        const T *kp = current_k_base + s * head_dim;

        double score = 0.0f;
        for (int d = 0; d < head_dim; ++d)
        {
            score += (double)current_q[d] * (double)kp[d];
        }

        attn_scores[s] = score * scale_factor;
    }

    if (is_causal)
    {
        for (int s = 0; s < src_seq_len; ++s)
        {
            if (s > target_idx)
            {
                attn_scores[s] = -CUDART_INF_F;
            }
        }
    }

    

    softmax(attn_scores, src_seq_len);

    for (int d = 0; d < head_dim; ++d)
    {
        double output_val = 0.0f;
        for (int s = 0; s < src_seq_len; ++s)
        {
            const T *vp = current_v_base + s * head_dim;
            output_val += attn_scores[s] * (double)vp[d];
        }
        current_o[d] = output_val;
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

    std::vector<double> hk, hv, hq, ho;

    for(auto x:h_k)hk.push_back(static_cast<double>(x));
    for(auto x:h_q)hq.push_back(static_cast<double>(x));
    for(auto x:h_v)hv.push_back(static_cast<double>(x));
    ho.resize(h_o.size());

    double *d_q, *d_k, *d_v,*d_o;

    CUDA_CHECK(cudaMalloc(&d_q, q_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_k, k_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v, v_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_o, o_size * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_q, hq.data(), q_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, hk.data(), k_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, hv.data(), v_size * sizeof(double), cudaMemcpyHostToDevice));

    int threads_per_block = 1024;

    dim3 block_size(threads_per_block);
    dim3 grid_size(
        (target_seq_len + threads_per_block - 1) / threads_per_block, // x: target sequence length
        query_heads,                                                  // y: query heads
        batch_size                                                    // z: batch size
    );

    flashAttentionKernel<<<grid_size, block_size>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim, is_causal);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(ho.data(), d_o, o_size * sizeof(double), cudaMemcpyDeviceToHost));

    for(int i=0;i<h_o.size();i++)
    {
        h_o[i]=float(ho[i]);
    }

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