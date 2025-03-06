#pragma once


template <typename scalar_t, uint tile_width>
__device__ void matmul_matrices(
    scalar_t* result_tile,
    
    const scalar_t* first_matrix,
    const scalar_t* second_matrix,

    uint tile_col,
    uint tile_row,
    
    const uint matrix_width,
    const uint matrix_shift
) {
    scalar_t first_matrix_tile_cache[tile_width];
    scalar_t second_matrix_tile_cache[tile_width];


    for (uint matrix_depth = 0; matrix_depth < matrix_width; ++matrix_depth) {
        // load just a partial row of the source matrix and a partial column of the inplace matrix
        for (uint load_step = 0; load_step < tile_width; ++load_step) {
            first_matrix_tile_cache[load_step] = first_matrix[tile_row];
            second_matrix_tile_cache[load_step] = second_matrix[tile_col];
            
            // load tile_width elements from left to right
            tile_row++;
            tile_col++;
        }
        // set to the next row
        tile_row += matrix_shift;
        tile_col += matrix_shift;
        
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