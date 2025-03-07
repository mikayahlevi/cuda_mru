#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/std/array>



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
__device__ inline void copy_matrix(const scalar_t* const source_matrices, scalar_t* const target_matrices, const uint matrix_size, const uint worker_idx, const uint total_workers) {
    typedef typename vector_type_info<scalar_t>::type vector_t;
    constexpr uint vector_type_size = vector_type_info<scalar_t>::elems_per_type;
    
    for (
        uint element_idx = worker_idx;
        element_idx < (matrix_size / vector_type_size);
        element_idx += total_workers
    ) {
        reinterpret_cast<vector_t*>(target_matrices)[element_idx] = reinterpret_cast<const vector_t*>(source_matrices)[element_idx];
    }
}


template <typename scalar_t, uint n_copies>
__device__ inline void copy_matrices(const cuda::std::array<const scalar_t* const, n_copies> source_matrices, const cuda::std::array<scalar_t* const, n_copies> target_matrices, const uint matrix_size, const uint worker_idx, const uint total_workers) {
    typedef typename vector_type_info<scalar_t>::type vector_t;
    constexpr uint vector_type_size = vector_type_info<scalar_t>::elems_per_type;
    
    for (
        uint element_idx = worker_idx;
        element_idx < (matrix_size / vector_type_size);
        element_idx += total_workers
    ) {
        #pragma unroll
        for (uint copy = 0; copy < n_copies; ++copy) {
            reinterpret_cast<vector_t*>(target_matrices[copy])[element_idx] = reinterpret_cast<const vector_t*>(source_matrices[copy])[element_idx];
        }
    }
}

template <typename scalar_t>
__device__ inline void copy_matrix_transposed(const scalar_t* const source_matrices, scalar_t* const target_matrices, const uint matrix_width, const uint matrix_size, const uint worker_idx, const uint total_workers) {
    typedef typename vector_type_info<scalar_t>::type vector_t;
    constexpr uint vector_type_size = vector_type_info<scalar_t>::elems_per_type;
    
    for (
        uint element_idx = vector_type_size * worker_idx;
        element_idx < matrix_size;
        element_idx += vector_type_size * total_workers
    ) {
        const vector_t tmp = reinterpret_cast<const vector_t*>(source_matrices)[element_idx / vector_type_size];
        
        const uint transposed_element_idx = (element_idx % matrix_width) * matrix_width + (element_idx / matrix_width);

        if constexpr (vector_type_size == 4) {
            target_matrices[transposed_element_idx] = tmp.x;
            target_matrices[transposed_element_idx + matrix_width] = tmp.y;
            target_matrices[transposed_element_idx + 2 * matrix_width] = tmp.z;
            target_matrices[transposed_element_idx + 3 * matrix_width] = tmp.w;
        } if constexpr (vector_type_size == 2) {
            target_matrices[transposed_element_idx] = tmp.x;
            target_matrices[transposed_element_idx + matrix_width] = tmp.y;
        }
    }
}


template <typename scalar_t, uint n_copies>
__device__ inline void copy_matrices_transposed(cuda::std::array<const scalar_t* const, n_copies> source_matrices, const cuda::std::array<scalar_t* const, n_copies> target_matrices, const uint matrix_width, const uint matrix_size, const uint worker_idx, const uint total_workers) {
    typedef typename vector_type_info<scalar_t>::type vector_t;
    constexpr uint vector_type_size = vector_type_info<scalar_t>::elems_per_type;
    
    for (
        uint element_idx = vector_type_size * worker_idx;
        element_idx < matrix_size;
        element_idx += vector_type_size * total_workers
    ) {
        #pragma unroll
        for (uint copy = 0; copy < n_copies; ++copy) {
            const vector_t tmp = reinterpret_cast<const vector_t*>(source_matrices[copy])[element_idx / vector_type_size];
            
            const uint transposed_element_idx = (element_idx % matrix_width) * matrix_width + (element_idx / matrix_width);

            if constexpr (vector_type_size == 4) {
                target_matrices[copy][transposed_element_idx] = tmp.x;
                target_matrices[copy][transposed_element_idx + matrix_width] = tmp.y;
                target_matrices[copy][transposed_element_idx + 2 * matrix_width] = tmp.z;
                target_matrices[copy][transposed_element_idx + 3 * matrix_width] = tmp.w;
            } if constexpr (vector_type_size == 2) {
                target_matrices[copy][transposed_element_idx] = tmp.x;
                target_matrices[copy][transposed_element_idx + matrix_width] = tmp.y;
            }
        }
    }
}