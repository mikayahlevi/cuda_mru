#include "common/info.cuh"
#include "common/write_tile.cuh"
#include "common/matmul.cuh"
#include "common/copy_matrix.cuh"

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


#ifndef TILE_WIDTH
#error "TILE_WIDTH must be defined in the compiler flags"
#endif

#ifndef MAX_MATMULS_PER_BLOCK
#error "MAX_MATMULS_PER_BLOCK must be defined in the compiler flags"
#endif


template <typename scalar_t, uint tile_width, uint tile_size>
__global__ void mru_cuda_forward_scan_stage_kernel(
    scalar_t* states,
    
    const uint scan_stage,

    const uint scan_stage_offset,

    const uint state_row_size,
    const uint state_matrix_size,
    const uint state_sequence_size,

    const uint n_source_matrices,
    const uint n_inplace_matrices,

    const uint tiled_state_width,
    const uint tiled_state_size,

    const uint matmuls_per_block
) {
    // blockIdx.x is the batch index
    // advance state pointer to the correct batch
    states += state_sequence_size * blockIdx.x;


    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem_cache_raw[];
    #define smem_cache reinterpret_cast<scalar_t*>(smem_cache_raw)

    // first n_source_matrices * state_matrix_size elements are part of source_matrix_smem_cache
    // n_source_matrices long
    scalar_t* const source_matrices_smem_cache = smem_cache;
    // rest of the elements are part of inplace_matrices_smem_cache
    // n_inplace_matrices long
    scalar_t* const inplace_matrices_smem_cache = &smem_cache[n_source_matrices * state_matrix_size];
    


    // tiled_state_size is the # threads per matmul

    // threads working on the same matrix are consecutive
    const uint intra_block_matmul_idx = threadIdx.x / tiled_state_size;
    const uint matmul_idx = blockIdx.y * matmuls_per_block + intra_block_matmul_idx;
    const uint tile_idx = threadIdx.x % tiled_state_size;

    // resolve indices

    const uint matmul_input_group = matmul_idx / scan_stage_offset;
    const uint matmul_input_group_idx = scan_stage_offset + 2 * scan_stage_offset * matmul_input_group;
    const uint matmul_input_group_subidx = matmul_idx % scan_stage_offset;


    /*
    global indices are different then shared indices because the shared
    indices exclude matrices that are not loaded into the smem for the current block 
    */

    const uint source_matrix_gmem_idx = matmul_input_group_idx - 1;
    const uint inplace_matrix_gmem_idx = matmul_input_group_idx + matmul_input_group_subidx;


    const uint source_matrix_smem_idx = intra_block_matmul_idx / scan_stage_offset;
    const uint inplace_matrix_smem_idx = intra_block_matmul_idx;


    // point to the correct memory locations
    // threads to matrix location is a many-to-one function
    const scalar_t* const source_matrix_gmem_ptr = &states[state_matrix_size * source_matrix_gmem_idx];
    /* */ scalar_t* const inplace_matrix_gmem_ptr = &states[state_matrix_size * inplace_matrix_gmem_idx];

    // not const because it will be loaded in
    scalar_t* const source_matrix_smem_ptr = &source_matrices_smem_cache[state_matrix_size * source_matrix_smem_idx];
    scalar_t* const inplace_matrix_smem_ptr = &inplace_matrices_smem_cache[state_matrix_size * inplace_matrix_smem_idx];




    // make the equations more intuitive for the gmem reads and writes
    #define threads_per_matmul tiled_state_size
    #define intra_matmul_thread_idx tile_idx


    const uint intra_block_input_group_subidx = matmul_input_group_subidx % matmuls_per_block;


    // load the source matrix into smem
    __syncthreads();
    {
        const uint matmuls_per_source_matrix = min(matmuls_per_block, scan_stage_offset);
        const uint threads_per_source_matrix = threads_per_matmul * matmuls_per_source_matrix;

        copy_matrix_transposed<scalar_t>(
            source_matrix_gmem_ptr,
            source_matrix_smem_ptr,
            
            state_row_size,
            state_matrix_size,
            
            intra_matmul_thread_idx + threads_per_matmul * intra_block_input_group_subidx,
            threads_per_source_matrix
        );
    }


    // load the inplace matrix into smem
    __syncthreads();
    copy_matrix<scalar_t>(inplace_matrix_gmem_ptr, inplace_matrix_smem_ptr, state_matrix_size, intra_matmul_thread_idx, threads_per_matmul);
    
    // compute the matmuls
    scalar_t result_tile[tile_size] = {0.0};
    
    // coords of the top left corner of the tile
    const uint tile_row = tile_width * (tile_idx / tiled_state_width);
    const uint tile_col = tile_width * (tile_idx % tiled_state_width);

    const uint tile_depth_matmul_offset = state_row_size * (n_source_matrices * (tile_idx / 32) + intra_block_input_group_subidx);

    __syncthreads();
    matmul_matrices<scalar_t, tile_width>(
        result_tile,

        source_matrix_smem_ptr,
        inplace_matrix_smem_ptr,

        tile_col + tile_depth_matmul_offset,
        tile_row + tile_depth_matmul_offset,

        state_matrix_size,
        state_row_size
    );

    __syncthreads();
    write_tile<scalar_t, tile_width, false>(result_tile, inplace_matrix_smem_ptr, tile_col, tile_row, state_row_size);


    // write results from smem back to gmem
    __syncthreads();
    copy_matrix<scalar_t>(inplace_matrix_smem_ptr, inplace_matrix_gmem_ptr, state_matrix_size, intra_matmul_thread_idx, threads_per_matmul);

    #undef threads_per_matmul
    #undef intra_matmul_thread_idx
}



void mru_cuda_forward(
    torch::Tensor states,

    const uint batch_size,
    const uint sequence_length,
    const uint state_width
) {
    const mru_scan_info scan_info = get_scan_info<TILE_WIDTH, MAX_MATMULS_PER_BLOCK>(state_width, sequence_length);
    const mru_general_info general_info = get_general_info<TILE_WIDTH>(state_width, sequence_length);
    
    dim3 scan_grid_dims(batch_size, scan_info.blocks_per_sequence, 1);

    AT_DISPATCH_FLOATING_TYPES(states.scalar_type(), "mru_cuda_forward", ([&] {
        for (uint scan_stage = 0; scan_stage < scan_info.total_scan_stages; ++scan_stage) {
            const mru_stage_info stage_info = get_stage_info(scan_info, scan_stage);

            // matmuls_per_block source matrices, plus ceil_div(matmuls_per_block, 1 << scan_stage) inplace matrices
            const uint n_smem_elements = (stage_info.n_source_matrices + stage_info.n_inplace_matrices) * general_info.state_matrix_size;

            
            mru_cuda_forward_scan_stage_kernel
            <scalar_t, TILE_WIDTH, TILE_WIDTH * TILE_WIDTH>
            <<<scan_grid_dims, scan_info.threads_per_block, n_smem_elements * sizeof(scalar_t)>>>
            (
                states.data_ptr<scalar_t>(),
                
                stage_info.scan_stage,
                
                stage_info.scan_stage_offset,

                general_info.state_row_size,
                general_info.state_matrix_size,
                general_info.state_sequence_size,

                stage_info.n_source_matrices,
                stage_info.n_inplace_matrices,

                general_info.tiled_state_width,
                general_info.tiled_state_size,

                scan_info.matmuls_per_block
            );
        }
    }));
}