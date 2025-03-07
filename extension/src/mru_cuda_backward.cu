#include "common/info.cuh"
#include "common/write_tile.cuh"
#include "common/matmul.cuh"
#include "common/copy_matrix.cuh"

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/std/array>


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


    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem_cache_raw[];
    #define smem_cache reinterpret_cast<scalar_t*>(smem_cache_raw)
    
    
    // optimize offsets to not be recalculated?
    // first n_source_matrices * state_matrix_size elements are part of source_matrix_smem_cache
    // n_source_matrices long
    scalar_t* const initial_states_source_matrices_smem_cache = smem_cache;
    scalar_t* const states_grad_source_matrices_smem_cache = &smem_cache[n_source_matrices * state_matrix_size];
    // rest of the elements are part of inplace_matrices_smem_cache
    // n_inplace_matrices long
    scalar_t* const initial_states_inplace_matrices_smem_cache = &smem_cache[2 * n_source_matrices * state_matrix_size];
    scalar_t* const states_grad_inplace_matrices_smem_cache = &smem_cache[2 * n_source_matrices * state_matrix_size + n_inplace_matrices * state_matrix_size];
    


    // threads working on the same matrix are consecutive
    const uint intra_block_matmul_idx = threadIdx.x / tiled_state_size;
    const uint matmul_idx = blockIdx.y * matmuls_per_block + intra_block_matmul_idx;
    const uint tile_idx = threadIdx.x % tiled_state_size;

    // resolve indices

    const uint matmul_input_group = matmul_idx / scan_stage_offset;
    const uint matmul_input_group_idx = scan_stage_offset + 2 * scan_stage_offset * matmul_input_group;
    const uint matmul_input_group_subidx = matmul_idx % scan_stage_offset;


    /*
    global indices are different then shared indices because the global
    indices include matrices that are irrelevant to the current block 
    */

    const uint source_matrix_gmem_idx = (sequence_length - 1) - (matmul_input_group_idx - 1);
    const uint inplace_matrix_gmem_idx = (sequence_length - 1) - (matmul_input_group_idx + matmul_input_group_subidx);


    const uint source_matrix_smem_idx = (n_source_matrices - 1) - (intra_block_matmul_idx / scan_stage_offset);
    const uint inplace_matrix_smem_idx = (n_inplace_matrices - 1) - intra_block_matmul_idx;


    // point to the correct memory locations
    // threads to matrix location is a many-to-one function
    const scalar_t* const initial_states_source_matrix_gmem_ptr = &initial_states[state_matrix_size * source_matrix_gmem_idx];
    const scalar_t* const states_grad_source_matrix_gmem_ptr = &states_grad[state_matrix_size * source_matrix_gmem_idx];
    /* */ scalar_t* const initial_states_inplace_matrix_gmem_ptr = &initial_states[state_matrix_size * inplace_matrix_gmem_idx];
    /* */ scalar_t* const states_grad_inplace_matrix_gmem_ptr = &states_grad[state_matrix_size * inplace_matrix_gmem_idx];

    // not const because it will be loaded in
    scalar_t* const initial_states_source_matrix_smem_ptr = &initial_states_source_matrices_smem_cache[state_matrix_size * source_matrix_smem_idx];
    scalar_t* const states_grad_source_matrix_smem_ptr = &states_grad_source_matrices_smem_cache[state_matrix_size * source_matrix_smem_idx];
    scalar_t* const initial_states_inplace_matrix_smem_ptr = &initial_states_inplace_matrices_smem_cache[state_matrix_size * inplace_matrix_smem_idx];
    scalar_t* const states_grad_inplace_matrix_smem_ptr = &states_grad_inplace_matrices_smem_cache[state_matrix_size * inplace_matrix_smem_idx];




    // make the equations more intuitive for the gmem reads and writes
    #define threads_per_matmul tiled_state_size
    #define intra_matmul_thread_idx tile_idx



    // load the source matrices into smem
    __syncthreads();
    {
        const uint intra_block_input_group_subidx = matmul_input_group_subidx % matmuls_per_block;
        const uint matmuls_per_source_matrix = min(matmuls_per_block, scan_stage_offset);
        const uint threads_per_source_matrix = threads_per_matmul * matmuls_per_source_matrix;

        copy_matrices_transposed<scalar_t, 2>(
            cuda::std::array<const scalar_t* const, 2>{initial_states_source_matrix_gmem_ptr, states_grad_source_matrix_gmem_ptr},
            cuda::std::array<scalar_t* const, 2>{initial_states_source_matrix_smem_ptr, states_grad_source_matrix_smem_ptr},
            state_row_size,
            state_matrix_size,
            intra_matmul_thread_idx + threads_per_matmul * intra_block_input_group_subidx,
            threads_per_source_matrix
        );
    }


    // load the inplace matrices into smem
    __syncthreads();
    copy_matrices<scalar_t, 2>(
        cuda::std::array<const scalar_t* const, 2>{initial_states_inplace_matrix_gmem_ptr, states_grad_inplace_matrix_gmem_ptr},
        cuda::std::array<scalar_t* const, 2>{initial_states_inplace_matrix_smem_ptr, states_grad_inplace_matrix_smem_ptr},
        state_matrix_size,
        intra_matmul_thread_idx,
        threads_per_matmul
    );

    
    // compute the matmuls
    {
        const uint new_intra_block_matmul_idx = threadIdx.x % matmuls_per_block;
    
        const uint new_source_matrix_smem_idx = (n_source_matrices - 1) - (new_intra_block_matmul_idx / scan_stage_offset);
        const uint new_inplace_matrix_smem_idx = (n_inplace_matrices - 1) - new_intra_block_matmul_idx;
    
        const scalar_t* const new_initial_states_source_matrix_smem_ptr = &initial_states_source_matrices_smem_cache[state_matrix_size * new_source_matrix_smem_idx];
        const scalar_t* const new_states_grad_source_matrix_smem_ptr = &states_grad_source_matrices_smem_cache[state_matrix_size * new_source_matrix_smem_idx];
    
        /* */ scalar_t* const new_initial_states_inplace_matrix_smem_ptr = &initial_states_inplace_matrices_smem_cache[state_matrix_size * new_inplace_matrix_smem_idx];
        /* */ scalar_t* const new_states_grad_inplace_matrix_smem_ptr = &states_grad_inplace_matrices_smem_cache[state_matrix_size * new_inplace_matrix_smem_idx];
    
    
        scalar_t result_tile[tile_size] = {0.0};
    
    
        const uint tile_idx = threadIdx.x / matmuls_per_block;
    
        // coords of the top left corner of the tile
        const uint tile_col = tile_width * (tile_idx % tiled_state_width);
        const uint tile_row = tile_width * (tile_idx / tiled_state_width);
    
        // move to the next row, then shift back tile_width to reset all the way to the left
        const uint matrix_shift = state_row_size - tile_width;
    
        __syncthreads();
        matmul_matrices<scalar_t, tile_width>(result_tile, new_states_grad_source_matrix_smem_ptr, new_initial_states_inplace_matrix_smem_ptr, tile_col, tile_row, state_row_size, matrix_shift);
    
        __syncthreads();
        write_tile<scalar_t, tile_width, true>(result_tile, new_states_grad_inplace_matrix_smem_ptr, tile_col, tile_row, state_row_size, matrix_shift);
    
        // reset result_tile to zeros
        for (uint i = 0; i < tile_size; ++i) result_tile[i] = 0.0;
    
        __syncthreads();
        matmul_matrices<scalar_t, tile_width>(result_tile, new_initial_states_source_matrix_smem_ptr, new_initial_states_inplace_matrix_smem_ptr, tile_col, tile_row, state_row_size, matrix_shift);
    
        __syncthreads();
        write_tile<scalar_t, tile_width, false>(result_tile, new_initial_states_inplace_matrix_smem_ptr, tile_col, tile_row, state_row_size, matrix_shift);
    }

    // write results from smem back to gmem
    __syncthreads();
    copy_matrices<scalar_t, 2>(
        cuda::std::array<const scalar_t* const, 2>{initial_states_inplace_matrix_smem_ptr, states_grad_inplace_matrix_smem_ptr},
        cuda::std::array<scalar_t* const, 2>{initial_states_inplace_matrix_gmem_ptr, states_grad_inplace_matrix_gmem_ptr},
        state_matrix_size,
        intra_matmul_thread_idx,
        threads_per_matmul
    );

    #undef threads_per_matmul
    #undef intra_matmul_thread_idx

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
    copy_matrices<scalar_t, 2>(
        cuda::std::array<const scalar_t* const, 2>{final_state_gmem_ptr, states_grad_gmem_ptr},
        cuda::std::array<scalar_t* const, 2>{final_states_smem_ptr, states_grad_smem_ptr},
        state_matrix_size,
        thread_idx,
        tiled_state_size
    );

    scalar_t result_tile[tile_size] = {0.0};

    // coords of the top left corner of the tile
    const uint tile_col = tile_width * (thread_idx % tiled_state_width);
    const uint tile_row = tile_width * (thread_idx / tiled_state_width);
    
    // move to the next row, then shift back tile_width to reset all the way to the left
    const uint matrix_shift = state_row_size - tile_width;

    __syncthreads();
    matmul_matrices<scalar_t, tile_width>(result_tile, final_states_smem_ptr, states_grad_smem_ptr, tile_col, tile_row, state_row_size, matrix_shift);

    __syncthreads(); 
    write_tile<scalar_t, tile_width, false>(result_tile, states_grad_smem_ptr, tile_col, tile_row, state_row_size, matrix_shift);

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
    constexpr uint tile_width = 4;
    constexpr uint tile_size = tile_width * tile_width;
    constexpr uint max_matmuls_per_block = 8;


    const mru_scan_info scan_info = get_scan_info<tile_width, max_matmuls_per_block>(state_width, sequence_length);
    const mru_general_info general_info = get_general_info<tile_width>(state_width, sequence_length);

    dim3 scan_grid_dims(batch_size, scan_info.blocks_per_sequence, 1);
    
    
    dim3 combine_grid_dims(batch_size, (sequence_length - 1), 1);

    AT_DISPATCH_FLOATING_TYPES(initial_states.scalar_type(), "mru_cuda_backward", ([&] {
        for (uint scan_stage = 0; scan_stage < scan_info.total_scan_stages; ++scan_stage) {
            const mru_stage_info stage_info = get_stage_info(scan_info, scan_stage);

            const uint n_smem_elements = 2 * (stage_info.n_source_matrices + stage_info.n_inplace_matrices) * general_info.state_matrix_size; 
            

            mru_cuda_backward_scan_stage_kernel
            <scalar_t, tile_width, tile_size>
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
        <scalar_t, tile_width, tile_size>
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