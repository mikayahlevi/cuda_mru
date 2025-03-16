#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, uint tile_width, bool accumulate>
__device__ inline void write_tile(
    const scalar_t* result_tile,
    scalar_t* target_matrix_tile,

    const uint tile_col,
    const uint tile_row,

    const uint matrix_width
) {
    // index of the top left corner of the tile
    uint store_idx = tile_row * matrix_width + tile_col;
    uint result_tile_idx = 0;

    for (uint row_idx = 0; row_idx < tile_width; ++row_idx) {
        #pragma unroll
        for (uint col_idx = 0; col_idx < tile_width; ++col_idx) {
            if constexpr (accumulate) {
                target_matrix_tile[store_idx + col_idx] += result_tile[result_tile_idx + col_idx];
            } else {
                target_matrix_tile[store_idx + col_idx]  = result_tile[result_tile_idx + col_idx];
            }
        }

        store_idx += matrix_width;
        result_tile_idx += tile_width;
    }
}