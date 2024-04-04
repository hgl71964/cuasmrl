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
        # ['LDG', 'STG', 'LDS', 'LDSM'],  # memory_ops
        ['LDG', 'STG'],
        # ['LDGDEPBAR', 'DEPBAR', 'LDGSTS', 'EXIT', 'BAR.SYNC'],  # ban_ops
        ['LDGDEPBAR', 'DEPBAR', 'EXIT', 'BAR.SYNC', 'IADD3.X'],  # ban_ops
    ),
    (7, 5): (
        # memory_ops
        [
            'LDG',
            'LDS',
            'STG',
        ],
        # ban_ops
        [
            'ERRBAR',
            'MEMBAR',
            'BAR',
            'DEPBAR',
            'ULDC',
            'EXIT',
            'BAR.SYNC',
            'LDSM',
        ],
    ),
}


def get_mutatable_ops(cc):
    if cc in MUTATABLE_OPS:
        return MUTATABLE_OPS[cc]
    else:
        raise RuntimeError(f'unsupported compute capability: {cc}')


def get_min_stall_count(cc, opcode):
    if cc == (7, 5):
        return 7
    elif cc == (8, 0):
        if opcode.startswith('LDGSTS'):
            return 10
        elif opcode.startswith('STG'):
            return 10
        else:
            return 7
    else:
        raise RuntimeError(f'unsupported compute capability: {cc}')


def get_all_checklist(cc, opcode, dst, src):
    if cc == (7, 5):
        if opcode.startswith('CS2R'):
            return [dst] + src
        else:
            return src

    elif cc == (8, 0):
        return src
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
