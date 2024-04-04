import os
import torch
import pickle

from cuasmrl.utils.gpu_utils import get_gpu_name

COUNTER = 0


def save_data(
    bin,
    final_perf,
    init_perf,
    save_path,
) -> str:
    data = {}
    data['cubin'] = bin.asm['cubin']  # binary
    data['final_perf'] = final_perf
    data['init_perf'] = init_perf

    kernel_name = bin.metadata['name'][:20]
    file_name = f'{kernel_name}_{COUNTER}.pkl'

    full_path = os.path.join(save_path, file_name)
    with open(full_path, 'wb') as f:
        pickle.dump(data, f)

    COUNTER += 1
    return full_path


def read_data(path, ):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data
