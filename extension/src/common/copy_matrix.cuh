#pragma once

#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t> struct vector_type_info;
template <> struct vector_type_info<float> {
    typedef float4 type;
    static constexpr uint elems_per_type = 4;
};

template <> struct vector_type_info<double> {
    typedef double2 type;
    static constexpr uint elems_per_type = 2;
};



template <typename scalar_t>
__device__ void copy_matrix(
    const scalar_t* const source_matrix,
    /* */ scalar_t* const target_matrix,

    const uint matrix_size,

    const uint worker_idx,
    const uint total_workers
) {
    typedef typename vector_type_info<scalar_t>::type vector_t;
    constexpr uint vector_type_size = vector_type_info<scalar_t>::elems_per_type;
    
    for (
        uint element_idx = worker_idx;
        element_idx < (matrix_size / vector_type_size);
        element_idx += total_workers
    ) {
        reinterpret_cast<vector_t*>(target_matrix)[element_idx] = reinterpret_cast<const vector_t*>(source_matrix)[element_idx];
    }
}

template <typename scalar_t>
__device__ void copy_matrix_transposed(
    const scalar_t* const source_matrix,
    /* */ scalar_t* const target_matrix,

    const uint matrix_width,
    const uint matrix_size,
    const uint worker_idx,
    const uint total_workers
) {
    typedef typename vector_type_info<scalar_t>::type vector_t;
    constexpr uint vector_type_size = vector_type_info<scalar_t>::elems_per_type;
    
    for (
        uint element_idx = vector_type_size * worker_idx;
        element_idx < matrix_size;
        element_idx += vector_type_size * total_workers
    ) {
        const vector_t tmp = reinterpret_cast<const vector_t*>(source_matrix)[element_idx / vector_type_size];
        
        const uint transposed_element_idx = (element_idx % matrix_width) * matrix_width + (element_idx / matrix_width);

        if constexpr (vector_type_size == 4) {
            target_matrix[transposed_element_idx] = tmp.x;
            target_matrix[transposed_element_idx + matrix_width] = tmp.y;
            target_matrix[transposed_element_idx + 2 * matrix_width] = tmp.z;
            target_matrix[transposed_element_idx + 3 * matrix_width] = tmp.w;
        } if constexpr (vector_type_size == 2) {
            target_matrix[transposed_element_idx] = tmp.x;
            target_matrix[transposed_element_idx + matrix_width] = tmp.y;
        }
    }
}


template <typename scalar_t>
__device__ void copy_two_matrices(
    const scalar_t* const first_source_matrix,
    const scalar_t* const second_source_matrix,
    /* */ scalar_t* const first_target_matrix,
    /* */ scalar_t* const second_target_matrix,

    const uint matrix_size,
    
    const uint worker_idx,
    const uint total_workers
) {
    typedef typename vector_type_info<scalar_t>::type vector_t;
    constexpr uint vector_type_size = vector_type_info<scalar_t>::elems_per_type;
    
    for (
        uint element_idx = worker_idx;
        element_idx < (matrix_size / vector_type_size);
        element_idx += total_workers
    ) {
        reinterpret_cast<vector_t*>(first_target_matrix)[element_idx] = reinterpret_cast<const vector_t*>(first_source_matrix)[element_idx];
        reinterpret_cast<vector_t*>(second_target_matrix)[element_idx] = reinterpret_cast<const vector_t*>(second_source_matrix)[element_idx];
    }
}

template <typename scalar_t>
__device__ void copy_two_matrices_transposed(
    const scalar_t* const first_source_matrix,
    const scalar_t* const second_source_matrix,
    /* */ scalar_t* const first_target_matrix,
    /* */ scalar_t* const second_target_matrix,
        
    const uint matrix_width,
    const uint matrix_size,
    
    const uint worker_idx,
    const uint total_workers
) {
    typedef typename vector_type_info<scalar_t>::type vector_t;
    constexpr uint vector_type_size = vector_type_info<scalar_t>::elems_per_type;
    
    vector_t tmp;

    for (
        uint element_idx = vector_type_size * worker_idx;
        element_idx < matrix_size;
        element_idx += vector_type_size * total_workers
    ) {
        const uint transposed_element_idx = (element_idx % matrix_width) * matrix_width + (element_idx / matrix_width);


        tmp = reinterpret_cast<const vector_t*>(first_source_matrix)[element_idx / vector_type_size];
        if constexpr (vector_type_size == 4) {
            first_target_matrix[transposed_element_idx] = tmp.x;
            first_target_matrix[transposed_element_idx + matrix_width] = tmp.y;
            first_target_matrix[transposed_element_idx + 2 * matrix_width] = tmp.z;
            first_target_matrix[transposed_element_idx + 3 * matrix_width] = tmp.w;
        } if constexpr (vector_type_size == 2) {
            first_target_matrix[transposed_element_idx] = tmp.x;
            first_target_matrix[transposed_element_idx + matrix_width] = tmp.y;
        }

        tmp = reinterpret_cast<const vector_t*>(second_source_matrix)[element_idx / vector_type_size];
        if constexpr (vector_type_size == 4) {
            second_target_matrix[transposed_element_idx] = tmp.x;
            second_target_matrix[transposed_element_idx + matrix_width] = tmp.y;
            second_target_matrix[transposed_element_idx + 2 * matrix_width] = tmp.z;
            second_target_matrix[transposed_element_idx + 3 * matrix_width] = tmp.w;
        } if constexpr (vector_type_size == 2) {
            second_target_matrix[transposed_element_idx] = tmp.x;
            second_target_matrix[transposed_element_idx + matrix_width] = tmp.y;
        }
    }
}