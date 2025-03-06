import torch

from .extension import mru_cuda_functions

class mru_cuda_class(torch.autograd.Function):
    @staticmethod
    def forward(ctx, initial_states):
        final_states = initial_states.clone()
        mru_cuda_functions.forward(final_states)
        ctx.save_for_backward(initial_states, final_states)
        return final_states

    @staticmethod
    def backward(ctx, grad_states):
        initial_states, final_states = ctx.saved_tensors
        
        inplace_initial_states = torch.empty_like(initial_states)
        inplace_initial_states[..., :-1, :, :].copy_(initial_states[..., 1:, :, :].transpose(-2, -1))
        inplace_initial_states[..., -1, :, :].fill_(0)

        inplace_grad_states = grad_states.clone()

        mru_cuda_functions.backward(inplace_initial_states, final_states, inplace_grad_states)

        return inplace_grad_states
        


op = mru_cuda_class.apply