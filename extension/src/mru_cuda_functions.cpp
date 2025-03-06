#include <torch/extension.h>
#include <cmath>
#include <cassert>



#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void mru_cuda_forward(
    torch::Tensor states,

    const uint batch_size,
    const uint sequence_length,
    const uint state_width
);


void mru_cuda_backward(
    torch::Tensor initial_states,
    torch::Tensor final_states,
    torch::Tensor states_grad,

    const uint batch_size,
    const uint sequence_length,
    const uint state_width
);


void launch_mru_cuda_forward(torch::Tensor states) {
    CHECK_INPUT(states);

    // resolve dimensions
    assert(states.size(-1) == states.size(-2)); // state matrices must be square
    const uint state_width = states.size(-1);

    const uint sequence_length = states.size(-3);

    const uint n_batch_dims = states.dim() - 3;


    // resolve batch size
    uint batch_size = 1;
    for (uint i = 0; i < n_batch_dims; i++) {
        batch_size *= states.size(i);
    }


    mru_cuda_forward(
        states,

        batch_size,
        sequence_length,
        state_width
    );
}


void launch_mru_cuda_backward(torch::Tensor initial_states, torch::Tensor final_states, torch::Tensor states_grad) {
    CHECK_INPUT(initial_states);
    CHECK_INPUT(states_grad);

    assert(initial_states.sizes() == final_states.sizes()); // all tensors must have the same shape
    assert(initial_states.sizes() == states_grad.sizes());

    // resolve dimensions
    assert(initial_states.size(-1) == initial_states.size(-2)); // state matrices must be square
    const uint state_width = initial_states.size(-1);

    const uint sequence_length = initial_states.size(-3);

    const uint n_batch_dims = initial_states.dim() - 3;

    // resolve batch size
    uint batch_size = 1;
    for (uint i = 0; i < n_batch_dims; i++) {
        batch_size *= initial_states.size(i);
    }

    // process gradients
    mru_cuda_backward(
        initial_states,
        final_states,
        states_grad,

        batch_size,
        sequence_length,
        state_width
    );
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_mru_cuda_forward, "inplace MRU CUDA forward implementation");
    m.def("backward", &launch_mru_cuda_backward, "inplace MRU CUDA backward implementation");
}