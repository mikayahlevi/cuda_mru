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
__global__ void mru_cuda_backward_scan_stage_kernel(
    scalar_t* initial_states,
    scalar_t* states_grad,

    const uint sequence_length,
    
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
    initial_states += state_sequence_size * blockIdx.x;
    states_grad += state_sequence_size * blockIdx.x;

    // idk why it needs to be allocated as an unsigned char but otherwise it doesn't compile
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem_cache_raw[];
    #define smem_cache reinterpret_cast<scalar_t*>(smem_cache_raw)
    
    
    /*
    the source matrices are the matrices on the left in the sklansky scan that are only read from
    and are used in multiple matrix multiplications per stage
    
    the inplace matrices are the matrices on the right that the results of the matrix multiplications are written to
    */
    // split the shared memory into two parts for the source and inplace matrices
    // split the cache for the inplace matrices into parts for initial_states and states_grad
    // source_matrices_smem_cache can be shared because the source matrices for initial_states and states_grad are never loaded concurrently
    scalar_t* const source_matrices_smem_cache = smem_cache;
    scalar_t* const initial_states_inplace_matrices_smem_cache = &smem_cache[n_source_matrices * state_matrix_size];
    scalar_t* const states_grad_inplace_matrices_smem_cache = &smem_cache[n_source_matrices * state_matrix_size + n_inplace_matrices * state_matrix_size];
    

    // tiled_state_size is the same as the number of threads per matrix multiplication
    // tile_idx is the thread index within the matrix multiplication

    const uint intra_block_matmul_idx = threadIdx.x / tiled_state_size;
    const uint matmul_idx = blockIdx.y * matmuls_per_block + intra_block_matmul_idx;
    const uint tile_idx = threadIdx.x % tiled_state_size;

    // resolve indices for a sklansky scan

    const uint sklansky_input_group = matmul_idx / scan_stage_offset;
    const uint sklansky_input_group_idx = scan_stage_offset + 2 * scan_stage_offset * sklansky_input_group;
    const uint sklansky_input_group_subidx = matmul_idx % scan_stage_offset;

    const uint intra_block_sklansky_input_group_subidx = sklansky_input_group_subidx % matmuls_per_block;

    /*
    the indices for global and shared memory are different because the shared memory only stores the ones that
    will be used for the current scan stage
    */

    const uint source_matrix_gmem_idx = (sequence_length - 1) - (sklansky_input_group_idx - 1);
    const uint inplace_matrix_gmem_idx = (sequence_length - 1) - (sklansky_input_group_idx + sklansky_input_group_subidx);

    const uint source_matrix_smem_idx = (n_source_matrices - 1) - (intra_block_matmul_idx / scan_stage_offset);
    const uint inplace_matrix_smem_idx = (n_inplace_matrices - 1) - intra_block_matmul_idx;


    // point to the correct memory locations
    const scalar_t* const initial_states_source_matrix_gmem_ptr = &initial_states[state_matrix_size * source_matrix_gmem_idx];
    const scalar_t* const states_grad_source_matrix_gmem_ptr = &states_grad[state_matrix_size * source_matrix_gmem_idx];
    /* */ scalar_t* const initial_states_inplace_matrix_gmem_ptr = &initial_states[state_matrix_size * inplace_matrix_gmem_idx];
    /* */ scalar_t* const states_grad_inplace_matrix_gmem_ptr = &states_grad[state_matrix_size * inplace_matrix_gmem_idx];

    // initial_states_source_matrix_smem_ptr and states_grad_source_matrix_smem_ptr can share the same location since they will not be loaded in concurrently
    scalar_t* const initial_states_source_matrix_smem_ptr = &source_matrices_smem_cache[state_matrix_size * source_matrix_smem_idx];
    scalar_t* const states_grad_source_matrix_smem_ptr    = &source_matrices_smem_cache[state_matrix_size * source_matrix_smem_idx];

    scalar_t* const initial_states_inplace_matrix_smem_ptr = &initial_states_inplace_matrices_smem_cache[state_matrix_size * inplace_matrix_smem_idx];
    scalar_t* const states_grad_inplace_matrix_smem_ptr = &states_grad_inplace_matrices_smem_cache[state_matrix_size * inplace_matrix_smem_idx];



    // load the first needed source matrices into smem (states_grad)
    // threads from multiple matrix multiplications can load the same source matrix
    __syncthreads();
    {
        const uint matmuls_per_source_matrix = min(matmuls_per_block, scan_stage_offset);
        const uint threads_per_source_matrix = tiled_state_size * matmuls_per_source_matrix;

        copy_matrix_transposed<scalar_t>(
            states_grad_source_matrix_gmem_ptr,
            states_grad_source_matrix_smem_ptr,

            state_row_size,
            state_matrix_size,

            tile_idx + tiled_state_size * intra_block_sklansky_input_group_subidx,
            threads_per_source_matrix
        );
    }


    // load the inplace matrices into smem
    __syncthreads();
    copy_two_matrices<scalar_t>(
        initial_states_inplace_matrix_gmem_ptr, states_grad_inplace_matrix_gmem_ptr,
        initial_states_inplace_matrix_smem_ptr, states_grad_inplace_matrix_smem_ptr,

        state_matrix_size,

        tile_idx,
        tiled_state_size
    );


    // allocate a thread-specific tile for the result of the matrix multiplication
    scalar_t result_tile[tile_size] = {0.0};
    
        
    // coords of the top left corner of the tile within the inplace matrix
    const uint tile_col = tile_width * (tile_idx % tiled_state_width);
    const uint tile_row = tile_width * (tile_idx / tiled_state_width);

    /*
    offset the start index for the the inner product of the matrix multiplication on a thread-wise basis so 
    that threads won't access the same memory at the same time, avoiding bank conflicts
    */
    const uint tile_depth_matmul_offset = state_row_size * (n_source_matrices * (tile_idx / 32) + intra_block_sklansky_input_group_subidx);

    __syncthreads();
    matmul_matrices<scalar_t, tile_width>(
        result_tile,

        states_grad_source_matrix_smem_ptr,
        initial_states_inplace_matrix_smem_ptr,

        tile_col + tile_depth_matmul_offset,
        tile_row + tile_depth_matmul_offset,

        state_matrix_size,
        state_row_size
    );

    // write back to the respective inplace matrix, which will be used as a buffer before writing back to global memory
    __syncthreads();
    write_tile<scalar_t, tile_width, true>(result_tile, states_grad_inplace_matrix_smem_ptr, tile_col, tile_row, state_row_size);

    // reset result_tile to zeros
    for (uint i = 0; i < tile_size; ++i) result_tile[i] = 0.0;


    // load the first needed source matrices into smem (intitial_states)
    __syncthreads();
    {
        const uint matmuls_per_source_matrix = min(matmuls_per_block, scan_stage_offset);
        const uint threads_per_source_matrix = tiled_state_size * matmuls_per_source_matrix;

        copy_matrix_transposed<scalar_t>(
            initial_states_source_matrix_gmem_ptr,
            initial_states_source_matrix_smem_ptr,

            state_row_size,
            state_matrix_size,

            tile_idx + tiled_state_size * intra_block_sklansky_input_group_subidx,
            threads_per_source_matrix
        );
    }


    __syncthreads();
    matmul_matrices<scalar_t, tile_width>(result_tile,
        initial_states_source_matrix_smem_ptr,
        initial_states_inplace_matrix_smem_ptr,

        tile_col + tile_depth_matmul_offset,
        tile_row + tile_depth_matmul_offset,

        state_matrix_size,
        state_row_size
    );

    // write back to the respective inplace matrix, which will be used as a buffer before writing back to global memory
    __syncthreads();
    write_tile<scalar_t, tile_width, false>(result_tile, initial_states_inplace_matrix_smem_ptr, tile_col, tile_row, state_row_size);


    // write results from shared memory back to global memory
    __syncthreads();
    copy_two_matrices<scalar_t>(
        initial_states_inplace_matrix_smem_ptr, states_grad_inplace_matrix_smem_ptr,
        initial_states_inplace_matrix_gmem_ptr, states_grad_inplace_matrix_gmem_ptr,

        state_matrix_size,

        tile_idx,
        tiled_state_size
    );

    #undef smem_cache
}


template <typename scalar_t, uint tile_width, uint tile_size>
__global__ void mru_backward_combine_kernel(
    const scalar_t* final_states,
    /* */ scalar_t* states_grad,

    const uint sequence_length,

    const uint state_row_size,
    const uint state_matrix_size,
    const uint state_sequence_size,

    const uint tiled_state_width,
    const uint tiled_state_size
) {
    // blockIdx.x is the batch index
    // advance state pointer to the correct batch
    final_states += state_sequence_size * blockIdx.x;
    states_grad += state_sequence_size * blockIdx.x;

    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem_cache_raw[];
    #define smem_cache reinterpret_cast<scalar_t*>(smem_cache_raw)

    // only effect [1:] of the sequence
    const uint final_state_matrix_idx = blockIdx.y;
    const uint states_grad_matrix_idx = blockIdx.y + 1;


    scalar_t* const final_states_smem_ptr = smem_cache;
    scalar_t* const states_grad_smem_ptr = &smem_cache[state_matrix_size];

    const scalar_t* const final_state_gmem_ptr = &final_states[final_state_matrix_idx * state_matrix_size];
    /* */ scalar_t* const states_grad_gmem_ptr = &states_grad[states_grad_matrix_idx * state_matrix_size];


    const uint thread_idx = threadIdx.x;

    // load the matrices into smem
    __syncthreads();
    copy_two_matrices<scalar_t>(
        final_state_gmem_ptr, states_grad_gmem_ptr,
        final_states_smem_ptr, states_grad_smem_ptr,

        state_matrix_size,

        thread_idx,
        tiled_state_size
    );

    scalar_t result_tile[tile_size] = {0.0};

    // coords of the top left corner of the tile
    const uint tile_col = tile_width * (thread_idx % tiled_state_width);
    const uint tile_row = tile_width * (thread_idx / tiled_state_width);

    __syncthreads();
    matmul_matrices<scalar_t, tile_width>(result_tile, final_states_smem_ptr, states_grad_smem_ptr, tile_col, tile_row, state_matrix_size, state_row_size);

    __syncthreads(); 
    write_tile<scalar_t, tile_width, false>(result_tile, states_grad_smem_ptr, tile_col, tile_row, state_row_size);

    __syncthreads();
    copy_matrix<scalar_t>(
        states_grad_smem_ptr,
        states_grad_gmem_ptr,

        state_matrix_size,

        thread_idx,
        tiled_state_size
    );

    #undef smem_cache
}


void mru_cuda_backward(
    torch::Tensor initial_states,
    torch::Tensor final_states,
    torch::Tensor states_grad,

    const uint batch_size,
    const uint sequence_length,
    const uint state_width
) {
    const mru_scan_info scan_info = get_scan_info<TILE_WIDTH, MAX_MATMULS_PER_BLOCK>(state_width, sequence_length);
    const mru_general_info general_info = get_general_info<TILE_WIDTH>(state_width, sequence_length);

    dim3 scan_grid_dims(batch_size, scan_info.blocks_per_sequence, 1);
    
    
    dim3 combine_grid_dims(batch_size, (sequence_length - 1), 1);

    AT_DISPATCH_FLOATING_TYPES(initial_states.scalar_type(), "mru_cuda_backward", ([&] {
        for (uint scan_stage = 0; scan_stage < scan_info.total_scan_stages; ++scan_stage) {
            const mru_stage_info stage_info = get_stage_info(scan_info, scan_stage);

            const uint n_smem_elements = (stage_info.n_source_matrices + 2 * stage_info.n_inplace_matrices) * general_info.state_matrix_size; 
            

            mru_cuda_backward_scan_stage_kernel
            <scalar_t, TILE_WIDTH, TILE_WIDTH * TILE_WIDTH>
            <<<scan_grid_dims, scan_info.threads_per_block, n_smem_elements * sizeof(scalar_t)>>>
            (
                initial_states.data_ptr<scalar_t>(),
                states_grad.data_ptr<scalar_t>(),

                sequence_length,

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


        mru_backward_combine_kernel
        <scalar_t, TILE_WIDTH, TILE_WIDTH * TILE_WIDTH>
        <<<combine_grid_dims, general_info.tiled_state_size, 2 * general_info.state_matrix_size * sizeof(scalar_t)>>>(
            final_states.data_ptr<scalar_t>(),
            states_grad.data_ptr<scalar_t>(),

            sequence_length,

            general_info.state_row_size,
            general_info.state_matrix_size,
            general_info.state_sequence_size,

            general_info.tiled_state_width,
            general_info.tiled_state_size
        );
    }));
}