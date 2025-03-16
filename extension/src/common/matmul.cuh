#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, uint tile_width>
__device__ void matmul_matrices(
    scalar_t* result_tile,
    
    const scalar_t* first_matrix,
    const scalar_t* second_matrix,

    uint tile_col,
    uint tile_row,
    
    const uint matrix_size,
    const uint matrix_width
) {
    scalar_t first_matrix_tile_cache[tile_width];
    scalar_t second_matrix_tile_cache[tile_width];


    for (uint matrix_depth = 0; matrix_depth < matrix_width; ++matrix_depth) {
        // allow both indices to wrap around
        tile_row %= matrix_size;
        tile_col %= matrix_size;

        // load just a partial row of the source matrix and a partial column of the inplace matrix
        #pragma unroll
        for (uint load_step = 0; load_step < tile_width; ++load_step) {
            first_matrix_tile_cache[load_step] = first_matrix[tile_row + load_step];
            second_matrix_tile_cache[load_step] = second_matrix[tile_col + load_step];
        }
        // set to the next row
        tile_row += matrix_width;
        tile_col += matrix_width;
        
        
        uint result_tile_idx = 0;
        // perform partial matrix multiplication on just the tile
        for (uint row_idx = 0; row_idx < tile_width; ++row_idx) {
            for (uint col_idx = 0; col_idx < tile_width; ++col_idx) {    
                result_tile[result_tile_idx] += first_matrix_tile_cache[row_idx] * second_matrix_tile_cache[col_idx];
                result_tile_idx++;
            }
        }
    }
} 