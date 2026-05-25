// Phase 2b: fused index_replacer + scatter kernel.
//
// In the BinnedKNNAutograd::forward path, the two-step
//   idx_unsorted = index_replacer(idx_sorted, sorting_indices)  // N*k allocation
//   idx_final.scatter_(0, sorting_indices.expand_as(idx_unsorted), idx_unsorted)
// holds N*k*8 bytes (idx_unsorted) and N*k*8 bytes (idx_final) simultaneously.
// At N=5M k=40 that's 2x 1.6 GB = 3.2 GB peak.
//
// This kernel computes the same result directly into idx_final with a
// single allocation, eliminating idx_unsorted:
//   idx_final[sorting_indices[i], j] = sorting_indices[idx_sorted[i, j]]
// with the same out-of-range / negative-index semantics as
// index_replacer_cuda_kernel.cu.
//
// Phase 2c: templated on the index dtype so the same kernel handles both
// int64 (canonical) and int32 (Phase 2c opt-in) idx_sorted inputs. The
// sorting_indices tensor stays int64 (it's produced by torch::argsort and
// PyTorch always returns int64 indices there).

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

template<typename IdxT>
__global__ void index_replacer_scatter_kernel(
    const IdxT* __restrict__ idx_sorted,           // [N, k]
    const int64_t* __restrict__ sorting_indices,   // [N]
    IdxT* __restrict__ idx_final,                  // [N, k]
    const int64_t N,
    const int64_t k
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * k;
    if (tid >= total) return;

    int64_t i = tid / k;
    int64_t j = tid % k;

    int64_t s = static_cast<int64_t>(idx_sorted[i * k + j]);
    int64_t orig_idx;
    if (s < 0) {
        orig_idx = s;  // pass-through (matches index_replacer semantics)
    } else if (s >= N) {
        // Matches the existing index_replacer fallback. In practice this
        // branch should not fire because sorting_indices values are in
        // [0, N), but preserve the semantics defensively.
        orig_idx = sorting_indices[i];
    } else {
        orig_idx = sorting_indices[s];
    }

    int64_t dest = sorting_indices[i];
    idx_final[dest * k + j] = static_cast<IdxT>(orig_idx);
}

torch::Tensor index_replacer_scatter_cuda_fn(
    torch::Tensor idx_sorted,         // [N, k] int32 or int64
    torch::Tensor sorting_indices     // [N] int64
) {
    CHECK_CUDA(idx_sorted);
    CHECK_CUDA(sorting_indices);
    TORCH_CHECK(idx_sorted.dtype() == torch::kInt64 || idx_sorted.dtype() == torch::kInt32,
                "idx_sorted must be int64 or int32");
    TORCH_CHECK(sorting_indices.dtype() == torch::kInt64, "sorting_indices must be int64");
    TORCH_CHECK(idx_sorted.dim() == 2, "idx_sorted must be 2D [N, k]");
    TORCH_CHECK(sorting_indices.dim() == 1, "sorting_indices must be 1D [N]");

    const int64_t N = idx_sorted.size(0);
    const int64_t k = idx_sorted.size(1);
    TORCH_CHECK(sorting_indices.size(0) == N, "sorting_indices size must equal idx_sorted.size(0)");

    auto idx_final = torch::empty_like(idx_sorted);
    const int64_t total = N * k;
    if (total == 0) {
        return idx_final;
    }

    const int64_t threads_per_block = 1024;
    const int64_t num_blocks = (total + threads_per_block - 1) / threads_per_block;

    if (idx_sorted.dtype() == torch::kInt64) {
        index_replacer_scatter_kernel<int64_t><<<num_blocks, threads_per_block>>>(
            idx_sorted.data_ptr<int64_t>(),
            sorting_indices.data_ptr<int64_t>(),
            idx_final.data_ptr<int64_t>(),
            N, k
        );
    } else {
        index_replacer_scatter_kernel<int32_t><<<num_blocks, threads_per_block>>>(
            idx_sorted.data_ptr<int32_t>(),
            sorting_indices.data_ptr<int64_t>(),
            idx_final.data_ptr<int32_t>(),
            N, k
        );
    }
    // Note: no explicit cudaDeviceSynchronize here; the caller synchronizes
    // before reading idx_final.
    return idx_final;
}
