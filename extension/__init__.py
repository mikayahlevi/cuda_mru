import torch.utils.cpp_extension

import os


parent_folder = os.path.dirname(os.path.abspath(__file__))

def get_path(relative_path: str):
    return str(os.path.join(parent_folder, relative_path))

mru_cuda_functions = torch.utils.cpp_extension.load(
    name = 'mru_cuda_functions',
    sources = [
        get_path('src/mru_cuda_functions.cpp'),
        get_path('src/mru_cuda_forward.cu'), get_path('src/mru_cuda_backward.cu')
    ],

    extra_include_paths = [get_path('src/common/')],
    
    extra_cuda_cflags = ['-DGTILE_WIDTH=4', '-DMAX_MATMULS_PER_BLOCK=8']
)

