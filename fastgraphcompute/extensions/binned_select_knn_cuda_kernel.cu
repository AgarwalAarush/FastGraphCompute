#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "binstepper.h"
#include "cuda_helpers.h"
#include "helpers.h"
#include <vector>
#include <string>
#include <cstdlib>
#include <c10/macros/Macros.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define C10_CUDA_KERNEL_LAUNCH_CHECK() {                         \
    cudaError_t err = cudaGetLastError();                        \
    if (err != cudaSuccess) {                                    \
        printf("CUDA Kernel launch error: %s\n",                 \
               cudaGetErrorString(err));                         \
        exit(EXIT_FAILURE);                                      \
    }                                                            \
}

__device__
static float calculateDistance(int64_t i_v, int64_t j_v, const float * d_coord, int64_t n_coords){
    float distsq=0;
    if(i_v == j_v)
        return 0;
    for(int64_t i=0;i<n_coords;i++){
        float dist = d_coord[I2D(i_v,i,n_coords)] - d_coord[I2D(j_v,i,n_coords)];
        distsq += dist*dist;
    }
    return distsq;
}


__device__ __forceinline__
static float calculateDistanceLocalChache(const float *  coord_i_v, int64_t j_v, const float * d_coord, int64_t n_coords){
    float distsq=0;
    for(int64_t i=0;i<n_coords;i++){
        float dist = coord_i_v[i] - d_coord[I2D(j_v,i,n_coords)];
        distsq += dist*dist;
    }
    return distsq;
}


__device__
static int64_t searchLargestDistance(int64_t i_v, float* d_dist, int64_t n_neigh, float& maxdist){

    maxdist=0;
    int64_t maxidx=0;
    if(n_neigh < 2)
        return maxidx;
    for(int64_t n=1;n<n_neigh;n++){ //0 is self
        float distsq = d_dist[I2D(i_v,n,n_neigh)];
        if(distsq > maxdist){
            maxdist = distsq;
            maxidx = n;
        }
    }
    return maxidx;
}


// Local-array version of searchLargestDistance. Reads from thread-local
// memory (or registers if the compiler keeps the array there) instead of
// the global d_dist/d_indices buffers. n_neigh is the runtime number of
// neighbours; K_MAX is the compile-time upper bound.
template<int K_MAX>
__device__ __forceinline__
static int local_search_largest_distance(const float* local_dist, int64_t n_neigh, float& maxdist){
    maxdist = 0.f;
    int maxidx = 0;
    #pragma unroll
    for(int n=1; n<K_MAX; n++){
        if(n >= (int)n_neigh) break;
        float d = local_dist[n];
        if(d > maxdist){
            maxdist = d;
            maxidx = n;
        }
    }
    return maxidx;
}


__global__
static void setDefaults(
        int64_t *d_indices,
        float *d_dist,
        const bool tf_compat,
        const int64_t n_vert,
        const int64_t n_neigh
){
    const int64_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert)
        return;
    const int64_t n =  blockIdx.y * blockDim.y + threadIdx.y;
    if(n >= n_neigh)
        return;

    int64_t idx = I2D(i_v,n,n_neigh);
    // Bounds check to prevent buffer overflow
    if(idx >= n_vert * n_neigh || idx < 0) {
        printf("CUDA setDefaults: Buffer overflow prevented at index %lld\n", idx);
        return;
    }

    if(n){
        if(tf_compat)
            d_indices[idx] = i_v;
        else
            d_indices[idx] = -1;
    }
    else{
        d_indices[idx] = i_v;
    }
    d_dist[idx] = 0;


}



template<int N_bin_dims, int K_MAX, typename T>
__global__
static void select_knn_kernel(

        const float * d_coord,
        const T * d_bin_idx,
        const T * d_direction,
        const T * d_dim_bin_idx,

        const T * d_bin_boundaries,
        const T * d_n_bins,

        const float* d_bin_width,

        int64_t *d_indices,
        float *d_dist,

        const int64_t n_vert,
        const int64_t n_neigh,
        const int64_t n_coords,
        const int64_t n_bin_dim,

        const int64_t n_bboundaries,
        bool use_direction) {

    //bin boundaries [i] [i+1] describe the scan ranges

    int64_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v>=n_vert)
        return;//safe guard

    // 0: can only be neighbour, 1: can only have neighbour, 2: neither
    if(use_direction &&
            (d_direction[i_v] == 0 || d_direction[i_v] == 2))
        return;

    // Per-thread top-K state held in compile-time-sized arrays. Compiler
    // either keeps these in registers (small K_MAX) or in L1-cached local
    // memory (larger K_MAX). Either way, this avoids the per-replacement
    // global rescan that the original kernel did against d_dist.
    //
    // Slot 0 is always self (i_v, distsq=0) and is written by setDefaults.
    // We track slots 1..K_MAX-1 here and write them back at kernel exit.
    float local_dist[K_MAX];
    int64_t local_idx[K_MAX];

    int64_t nfilled=1;//self-reference from defaults at slot 0
    int maxidx_local=0;
    float maxdistsq=0;

    int64_t total_subbins = 1;
    for(int64_t sbi=0;sbi<n_bin_dim;sbi++)
        total_subbins *= d_n_bins[sbi];

    int64_t iv_bin = d_bin_idx[i_v];
    int64_t gbin_offset = total_subbins*(iv_bin / total_subbins);
    int64_t sb_flat_offset = iv_bin - gbin_offset;

    float coord_i_v[10];//keep this and the next "10" in sync
    int64_t max_loc_n_coords = std::min(n_coords,(int64_t)10);
    for(int64_t i=0;i<max_loc_n_coords;i++){
        coord_i_v[i] = d_coord[I2D(i_v,i,n_coords)];
    }

    binstepper<N_bin_dims, T> stepper(d_n_bins, &d_dim_bin_idx[I2D(i_v,1,n_bin_dim+1)]);

    bool continue_search = true;
    int64_t distance = 0;
    while(continue_search){

        stepper.set_d(distance);

        continue_search=false;

        while(true){
            int64_t idx = stepper.step();
            if(idx<0){//not valid
                if(!continue_search && !distance){//this should not happen
                    printf("\nERROR: binned_select_knn.cu: stopping search for vtx %lld at distance %lld\n",i_v,distance);
                }
                break;

            }

            idx+=gbin_offset;

            if(idx>=n_bboundaries-1){
                printf("\nERROR: binned_select_knn.cu: boundary issue: idx %lld out of range, gb offset %lld, distance %lld, sb_flat_offset %lld, nbb %lld\n", idx, gbin_offset, distance, sb_flat_offset,n_bboundaries);
                continue;
            }

            int64_t start_vertex = d_bin_boundaries[idx];
            int64_t end_vertex = d_bin_boundaries[idx+1];

            if(start_vertex == end_vertex){ //empty bin
                continue_search=true; //correct?
                continue;
            }

            if(start_vertex>=n_vert || end_vertex>n_vert){
                printf("\nERROR: binned_select_knn.cu: start_vertex %lld or end_vertex %lld out of range %lld \n", start_vertex, end_vertex, n_vert);
                continue;//safe guard
            }

            for(int64_t j_v=start_vertex;j_v<end_vertex;j_v++){
                if(i_v == j_v)
                    continue;

                // 0: can only be neighbour, 1: can only have neighbour, 2: neither
                if(use_direction &&
                        (d_direction[j_v] == 1 || d_direction[j_v] == 2))
                    continue;

                //fill up
                float distsq = 0;
                if(max_loc_n_coords < n_coords)
                    distsq = calculateDistance(i_v,j_v,d_coord,n_coords);
                else
                    distsq = calculateDistanceLocalChache(coord_i_v,j_v,d_coord,n_coords);
                if(nfilled< n_neigh){
                    local_idx[nfilled]  = j_v;
                    local_dist[nfilled] = distsq;
                    if(distsq > maxdistsq){
                        maxdistsq = distsq;
                        maxidx_local = (int)nfilled;
                    }
                    nfilled++;
                    continue;
                }
                if(distsq < maxdistsq){// automatically applies to max radius
                    //replace former max in local arrays
                    local_idx[maxidx_local]  = j_v;
                    local_dist[maxidx_local] = distsq;
                    //search new max in local memory (no global rescan)
                    maxidx_local = local_search_largest_distance<K_MAX>(local_dist, n_neigh, maxdistsq);
                }
            }

            continue_search=true;//at least one was valid

        }
        if(nfilled==n_neigh && d_bin_width[0]*distance * d_bin_width[0]*distance > maxdistsq)
            break;//done

        distance++;
    }

    // Write filled slots back to global memory once. Slot 0 was written
    // by setDefaults; unfilled slots (nfilled..n_neigh-1) keep their
    // setDefaults values.
    for(int64_t n=1; n<nfilled; n++){
        d_indices[I2D(i_v,n,n_neigh)] = local_idx[n];
        d_dist[I2D(i_v,n,n_neigh)]    = local_dist[n];
    }
}


// Global-memory variant — top-K maintained directly in d_dist / d_indices.
// Used as a runtime fallback when (K > 64) or (n_coords >= 8): in that
// regime the per-thread local arrays in select_knn_kernel start spilling
// out of registers and the spill cost outweighs the saved global rescans.
// Algorithmically identical to the local-array kernel; the only
// difference is *where* the top-K state lives.
template<int N_bin_dims, typename T>
__global__
static void select_knn_kernel_global(

        const float * d_coord,
        const T * d_bin_idx,
        const T * d_direction,
        const T * d_dim_bin_idx,

        const T * d_bin_boundaries,
        const T * d_n_bins,

        const float* d_bin_width,

        int64_t *d_indices,
        float *d_dist,

        const int64_t n_vert,
        const int64_t n_neigh,
        const int64_t n_coords,
        const int64_t n_bin_dim,

        const int64_t n_bboundaries,
        bool use_direction) {

    int64_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v>=n_vert)
        return;

    if(use_direction &&
            (d_direction[i_v] == 0 || d_direction[i_v] == 2))
        return;

    int64_t nfilled=1;//self-reference from defaults at slot 0
    int64_t maxidx_local=0;
    float maxdistsq=0;

    int64_t total_subbins = 1;
    for(int64_t sbi=0;sbi<n_bin_dim;sbi++)
        total_subbins *= d_n_bins[sbi];

    int64_t iv_bin = d_bin_idx[i_v];
    int64_t gbin_offset = total_subbins*(iv_bin / total_subbins);
    int64_t sb_flat_offset = iv_bin - gbin_offset;

    float coord_i_v[10];
    int64_t max_loc_n_coords = std::min(n_coords,(int64_t)10);
    for(int64_t i=0;i<max_loc_n_coords;i++){
        coord_i_v[i] = d_coord[I2D(i_v,i,n_coords)];
    }

    binstepper<N_bin_dims, T> stepper(d_n_bins, &d_dim_bin_idx[I2D(i_v,1,n_bin_dim+1)]);

    bool continue_search = true;
    int64_t distance = 0;
    while(continue_search){
        stepper.set_d(distance);
        continue_search=false;

        while(true){
            int64_t idx = stepper.step();
            if(idx<0){
                if(!continue_search && !distance){
                    printf("\nERROR: binned_select_knn.cu (global): stopping search for vtx %lld at distance %lld\n",i_v,distance);
                }
                break;
            }

            idx+=gbin_offset;
            if(idx>=n_bboundaries-1){
                printf("\nERROR: binned_select_knn.cu (global): boundary issue: idx %lld out of range, gb offset %lld, distance %lld, sb_flat_offset %lld, nbb %lld\n", idx, gbin_offset, distance, sb_flat_offset,n_bboundaries);
                continue;
            }

            int64_t start_vertex = d_bin_boundaries[idx];
            int64_t end_vertex = d_bin_boundaries[idx+1];

            if(start_vertex == end_vertex){
                continue_search=true;
                continue;
            }

            if(start_vertex>=n_vert || end_vertex>n_vert){
                printf("\nERROR: binned_select_knn.cu (global): start_vertex %lld or end_vertex %lld out of range %lld\n", start_vertex, end_vertex, n_vert);
                continue;
            }

            for(int64_t j_v=start_vertex;j_v<end_vertex;j_v++){
                if(i_v == j_v)
                    continue;

                if(use_direction &&
                        (d_direction[j_v] == 1 || d_direction[j_v] == 2))
                    continue;

                float distsq = 0;
                if(max_loc_n_coords < n_coords)
                    distsq = calculateDistance(i_v,j_v,d_coord,n_coords);
                else
                    distsq = calculateDistanceLocalChache(coord_i_v,j_v,d_coord,n_coords);

                if(nfilled < n_neigh){
                    d_indices[I2D(i_v,nfilled,n_neigh)] = j_v;
                    d_dist[I2D(i_v,nfilled,n_neigh)] = distsq;
                    if(distsq > maxdistsq){
                        maxdistsq = distsq;
                        maxidx_local = nfilled;
                    }
                    nfilled++;
                    continue;
                }
                if(distsq < maxdistsq){
                    d_indices[I2D(i_v,maxidx_local,n_neigh)] = j_v;
                    d_dist[I2D(i_v,maxidx_local,n_neigh)] = distsq;
                    maxidx_local = searchLargestDistance(i_v,d_dist,n_neigh,maxdistsq);
                }
            }

            continue_search=true;
        }
        if(nfilled==n_neigh && d_bin_width[0]*distance * d_bin_width[0]*distance > maxdistsq)
            break;
        distance++;
    }
}


// Function to dispatch based on input tensor types (int64 or int64)
std::tuple<torch::Tensor, torch::Tensor> binned_select_knn_cuda_fn(
    torch::Tensor coordinates,
    torch::Tensor bin_idx,
    torch::Tensor dim_bin_idx,
    torch::Tensor bin_boundaries,
    torch::Tensor n_bins,
    torch::Tensor bin_width,
    torch::Tensor direction,
    bool tf_compat,
    bool use_direction,
    int64_t K
) {
    const auto n_vert = coordinates.size(0);
    const auto n_coords = coordinates.size(1);
    const auto n_bboundaries = bin_boundaries.size(0);
    const auto n_bin_dims = n_bins.size(0);
    const c10::cuda::CUDAGuard device_guard(coordinates.device());

    auto options_int = torch::TensorOptions().dtype(torch::kInt64).device(coordinates.device());
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(coordinates.device());

    torch::Tensor indices = torch::empty({n_vert, K}, options_int);
    torch::Tensor distances = torch::empty({n_vert, K}, options_float);

    grid_and_block gb_set_def(n_vert,256,K,4);
    grid_and_block gb(n_vert,512);
    auto stream = at::cuda::getCurrentCUDAStream();

    setDefaults<<<gb_set_def.grid(),gb_set_def.block(), 0, stream.stream()>>>(indices.data_ptr<int64_t>(), distances.data_ptr<float>(), tf_compat, n_vert, K);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    TORCH_CHECK(bin_idx.scalar_type() == torch::kInt64,
                "binned_select_knn_cuda: bin_idx must be int64.");

    // Tier-2a dispatch.
    //
    // Local-array kernel (select_knn_kernel) is faster in the regime
    // where the per-thread top-K state fits in registers / L1: low K
    // (≤64) and low coord dim (≤7). Outside that regime, the array
    // spills and the saved global rescans no longer cover the spill
    // cost; A/B benchmarks showed regressions of up to ~0.52x at
    // (dim=10, K=128). We fall back to the original global-memory
    // kernel in that regime — same algorithm, same neighbour set,
    // just stores top-K in d_dist/d_indices directly.
    //
    // Thresholds (K > 64) and (n_coords >= 8) come from measured
    // crossover, not a model. Override with the env var
    // FGC_FORCE_GLOBAL=1 to force the global kernel for everything
    // (useful for A/B testing or regression isolation).
    static const bool force_global = []{
        const char* e = std::getenv("FGC_FORCE_GLOBAL");
        return e && std::string(e) != "0" && !std::string(e).empty();
    }();
    const bool use_local = !force_global && (K <= 64) && (n_coords <= 7);

#define BSK_LAUNCH_LOCAL(NBD, KM)                                                           \
    select_knn_kernel<(NBD), (KM), int64_t><<<gb.grid(), gb.block(), 0, stream.stream()>>>( \
        coordinates.data_ptr<float>(), bin_idx.data_ptr<int64_t>(),                         \
        direction.data_ptr<int64_t>(), dim_bin_idx.data_ptr<int64_t>(),                     \
        bin_boundaries.data_ptr<int64_t>(), n_bins.data_ptr<int64_t>(),                     \
        bin_width.data_ptr<float>(), indices.data_ptr<int64_t>(),                           \
        distances.data_ptr<float>(), n_vert, K, n_coords, n_bin_dims, n_bboundaries,        \
        use_direction)

#define BSK_LAUNCH_GLOBAL(NBD)                                                              \
    select_knn_kernel_global<(NBD), int64_t><<<gb.grid(), gb.block(), 0, stream.stream()>>>(\
        coordinates.data_ptr<float>(), bin_idx.data_ptr<int64_t>(),                         \
        direction.data_ptr<int64_t>(), dim_bin_idx.data_ptr<int64_t>(),                     \
        bin_boundaries.data_ptr<int64_t>(), n_bins.data_ptr<int64_t>(),                     \
        bin_width.data_ptr<float>(), indices.data_ptr<int64_t>(),                           \
        distances.data_ptr<float>(), n_vert, K, n_coords, n_bin_dims, n_bboundaries,        \
        use_direction)

#define BSK_DISPATCH(NBD)                                                                   \
    do {                                                                                    \
        if (use_local) {                                                                    \
            if      (K <= 16) { BSK_LAUNCH_LOCAL(NBD, 16); }                                \
            else if (K <= 32) { BSK_LAUNCH_LOCAL(NBD, 32); }                                \
            else              { BSK_LAUNCH_LOCAL(NBD, 64); }                                \
        } else {                                                                            \
            BSK_LAUNCH_GLOBAL(NBD);                                                         \
        }                                                                                   \
    } while(0)

    if      (n_bin_dims == 2) BSK_DISPATCH(2);
    else if (n_bin_dims == 3) BSK_DISPATCH(3);
    else if (n_bin_dims == 4) BSK_DISPATCH(4);
    else if (n_bin_dims == 5) BSK_DISPATCH(5);
    else throw std::invalid_argument("Unsupported number of binning dimensions.");

#undef BSK_DISPATCH
#undef BSK_LAUNCH_GLOBAL
#undef BSK_LAUNCH_LOCAL

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(indices, distances);
}
