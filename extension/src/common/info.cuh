#pragma once

#include <cassert>


struct mru_general_info {
    const uint tiled_state_width;
    const uint tiled_state_size;

    const uint state_row_size;
    const uint state_matrix_size;
    const uint state_sequence_size;
};


struct mru_scan_info {
    const uint total_scan_stages;

    const uint matmuls_per_block;
    const uint matmuls_per_sequence;
    const uint blocks_per_sequence;
    const uint threads_per_block;
};

struct mru_stage_info {
    const uint scan_stage;
    const uint scan_stage_offset;

    const uint n_source_matrices;
    const uint n_inplace_matrices;
};

static constexpr uint ceil_div(const uint dividend, const uint divisor) {
    return (dividend + divisor - 1) / divisor;
}

static constexpr uint ceil_log2(uint x) {
    uint y = 0;
    while ((1 << y) < x) {
        y++;
    }
    return y;
}

template <uint tile_width>
inline mru_general_info get_general_info(const uint state_width, const uint sequence_length) {
    const uint tiled_state_width = ceil_div(state_width, tile_width);
    const uint tiled_state_size = tiled_state_width * tiled_state_width;

    const uint state_row_size = state_width;
    const uint state_matrix_size = state_row_size * state_row_size;
    const uint state_sequence_size = state_matrix_size * sequence_length;
    
    return {
        .tiled_state_width = tiled_state_width,
        .tiled_state_size = tiled_state_size,

        .state_row_size = state_row_size,
        .state_matrix_size = state_matrix_size,
        .state_sequence_size = state_sequence_size
    };
}


// I still need to decrease the matmuls per block if the matrices are two big for a single block
template <uint tile_width, uint max_matmuls_per_block>
inline mru_scan_info get_scan_info(const uint state_width, const uint sequence_length) {
    assert(state_width % tile_width == 0); // state width must be divisible by the tile width
    // temporary restriction, may be removed in the future
    assert(sequence_length == (1 << ceil_log2(sequence_length))); // sequence length must be a power of 2


    const uint total_scan_stages = ceil_log2(sequence_length);


    const uint matmuls_per_block = std::min(ceil_div(sequence_length, 2), max_matmuls_per_block);
    assert(matmuls_per_block == (1 << ceil_log2(matmuls_per_block))); // matmuls_per_block must be a power of 2
    
    const uint matmuls_per_sequence = ceil_div(sequence_length, 2);
    const uint blocks_per_sequence = ceil_div(matmuls_per_sequence, matmuls_per_block);

    const uint tiled_state_width = ceil_div(state_width, tile_width);
    const uint tiled_state_size = tiled_state_width * tiled_state_width;

    const uint state_row_size = state_width;
    const uint state_matrix_size = state_row_size * state_row_size;
    const uint state_sequence_size = state_matrix_size * sequence_length;

    const uint threads_per_block = tiled_state_size * matmuls_per_block;

    return {
        .total_scan_stages = total_scan_stages,
        .matmuls_per_block = matmuls_per_block,
        .matmuls_per_sequence = matmuls_per_sequence,
        .blocks_per_sequence = blocks_per_sequence,
        .threads_per_block = threads_per_block
    };
}


inline mru_stage_info get_stage_info(const mru_scan_info scan_info, const uint scan_stage) {
    // calculate the offset for the scan
    const uint scan_stage_offset = 1 << scan_stage;
    const uint n_source_matrices = ceil_div(scan_info.matmuls_per_block, scan_stage_offset);
    const uint n_inplace_matrices = scan_info.matmuls_per_block;

    return {
        .scan_stage = scan_stage,
        .scan_stage_offset = scan_stage_offset,
        .n_source_matrices = n_source_matrices,
        .n_inplace_matrices = n_inplace_matrices
    };
}