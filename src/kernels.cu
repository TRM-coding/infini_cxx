#include <vector>
#include <iostream>
#include "../tester/utils.h"
#include <cassert>

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
            size_t mid = min(left_start + curr_size - 1, n - 1);
            size_t right_end = min(left_start + 2 * curr_size - 1, n - 1);

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
    const int device = 6;
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
    CUDA_CHECK(cudaGetLastError()); // 检查 launch
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(ans, cudaans, sizeof(T), cudaMemcpyDeviceToHost));
    return *ans;
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
__global__ void kernel(const T *h_q, const T *h_k, const T *h_v, T *h_o,
                       int batch_size, int head_dim, bool is_causal,
                       int query_heads, int kv_heads, int Brow, int Bcol)
{   
    int batch_idx=blockIdx.z/query_heads;
    int q_head_idx=blockIdx.z%query_heads;
    int group_size=hv_heads/query_heads;
    int kv_head_idx=q_head_idx/group_size;
    
}

template <typename T>
void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k,
                    const std::vector<T> &h_v, std::vector<T> &h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal)
{
    auto cile = [](int a, int b)
    { return (a + b - 1) / b; };
    int Brow = 32;
    int Bcol = 32;
    dim3 grid(cile(target_seq_len, Brow), cile(src_seq_len, Bcol), batch_size * query_heads);
    int block_size = Brow;
    T *h_q_c = nullptr;
    T *h_v_c = nullptr;
    T *h_k_c = nullptr;
    T *h_o_c = nullptr;
    cudaMalloc(&h_q_c, h_q.size() * sizeof(T));
    cudaMalloc(&h_v_c, h_v.size() * sizeof(T));
    cudaMalloc(&h_k_c, h_k.size() * sizeof(T));
    cudaMalloc(&h_o_c, h_o.size() * sizeof(T));

    cudaMemcpy(h_q_c, h_q.data(), cudaMemcpyHostToDevice);
    cudaMemcpy(h_k_c, h_k.data(), cudaMemcpyHostToDevice);
    cudaMemcpy(h_v_c, h_v.data(), cudaMemcpyHostToDevice);
    kernel<<<grid, block_size>>>(h_q_c, h_k_c, h_v_c, h_o_c, batch_size, head_dim, is_causal, query_heads,kv_heads, Brow, Bcol);
    cudaMemcpy(h_o.data(),h_o_c,cudaMemcpyDeviceToHost);
    return;

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
