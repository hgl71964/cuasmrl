import torch
import subprocess

# example sass opcode
# see : https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html

# adding GPU ISA probably need to add instruction repository
# https://github.com/cloudcores/CuAssembler/blob/master/UserGuide.md#instruction-assembler-repository

MUTATABLE_OPS = {
    (8, 9): (
        ['LDG', 'STG', 'LDS', 'LDSM'],  # memory_ops
        ['LDGDEPBAR', 'DEPBAR'],  # ban_ops
    ),
    (8, 0): (
        ['LDG', 'STG', 'LDS', 'LDSM'],  # memory_ops
        ['LDGDEPBAR', 'DEPBAR', 'LDGSTS', 'EXIT', 'BAR.SYNC'],  # ban_ops
    ),
    (7, 5): (
        ['LDG', 'LDS', 'LDSM'],  # memory_ops
        ['ERRBAR', 'MEMBAR', 'BAR', 'DEPBAR', 'ULDC'],  # ban_ops
    ),
}


def get_mutatable_ops(cc):
    if cc in MUTATABLE_OPS:
        return MUTATABLE_OPS[cc]
    else:
        raise RuntimeError(f'unsupported compute capability: {cc}')


def get_gpu_name():
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            universal_newlines=True)
        gpu_name = output.strip()
        gpu_name = gpu_name.replace(' ', '_')
        return gpu_name
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error: {e}")


def get_gpu_cc():
    if torch.cuda.is_available():
        # device = torch.device("cuda")
        # print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        compute_capability = torch.cuda.get_device_capability(0)
        # print(f"Compute Capability: {compute_capability}")
        return compute_capability  # e.g. (7, 5)
    else:
        raise RuntimeError("No GPU available, using CPU.")
