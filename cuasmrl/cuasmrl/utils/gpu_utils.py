import torch
import subprocess

# example sass opcode
# see : https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html

# adding GPU ISA probably need to add instruction repository
# https://github.com/cloudcores/CuAssembler/blob/master/UserGuide.md#instruction-assembler-repository

# deprecated
MUTATABLE_OPS = {
    # Ada; XXX not working
    (8, 9): (
        ['LDG', 'STG', 'LDS', 'LDSM'],  # memory_ops
        ['LDGDEPBAR', 'DEPBAR'],  # ban_ops
    ),
    # RTX3000
    (8, 6): (
        # ['LDG', 'STG', 'LDS', 'LDSM'],  # memory_ops
        ['LDG', 'STG'],
        # ['LDGDEPBAR', 'DEPBAR', 'LDGSTS', 'EXIT', 'BAR.SYNC'],  # ban_ops
        ['LDGDEPBAR', 'DEPBAR', 'EXIT', 'BAR.SYNC', 'IADD3.X'],  # ban_ops
    ),
    # A100
    (8, 0): (
        # ['LDG', 'STG', 'LDS', 'LDSM'],  # memory_ops
        ['LDGSTS', 'LDG', 'STG'],
        # ['LDGDEPBAR', 'DEPBAR', 'LDGSTS', 'EXIT', 'BAR.SYNC'],  # ban_ops
        ['LDGDEPBAR', 'DEPBAR', 'EXIT', 'BAR.SYNC', 'IADD3.X'],  # ban_ops
    ),
    # turing; RTX 8000
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
    # V100
    (7, 0): (
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
            # 'LDSM',
        ],
    ),
}


# deprecated
def get_mutatable_ops(cc):
    if cc in MUTATABLE_OPS:
        return MUTATABLE_OPS[cc]
    else:
        raise RuntimeError(f'unsupported compute capability: {cc}')


def is_mem_op(cc, opcode):
    if cc == (7, 5):
        if opcode.startswith('LDG'):
            return True
        elif opcode.startswith('STG'):
            return True
        elif opcode.startswith('LDS'):
            return True
        return False
    elif cc == (7, 0):
        if opcode.startswith('LDG'):
            return True
        elif opcode.startswith('STS'):
            return True
        elif opcode.startswith('LDS'):
            return True
        return False
    elif cc == (8, 0):

        # ban op
        if opcode.startswith('LDGDEPBAR'):
            return False

        # mem op
        elif opcode.startswith('LDGSTS'):
            return True
        elif opcode.startswith('LDG'):
            return True
        elif opcode.startswith('STG'):
            return True
        return False

    elif cc == (8, 6):
        # ban op
        if opcode.startswith('LDGDEPBAR'):
            return False

        # mem op
        elif opcode.startswith('LDGSTS'):
            return True
        elif opcode.startswith('LDG'):
            return True
        elif opcode.startswith('STG'):
            return True
        return False

    else:
        raise RuntimeError(f'unsupported compute capability: {cc}')


def get_min_stall_count(cc, opcode):
    if cc == (7, 5):
        if opcode.startswith('CS2R'):
            return 19
        return 7
    elif cc == (7, 0):
        if opcode.startswith('CS2R'):
            return 19
        elif opcode.startswith('LDG'):
            return 15
        return 7
    elif cc == (8, 0):
        return 8
    elif cc == (8, 6):
        return 8
    else:
        raise RuntimeError(f'unsupported compute capability: {cc}')


def get_moveup_deps(cc, opcode, tmp_dst, tmp_src, dst, src):
    # some opcodes needs both dst and src ready... so we need to check both
    if cc == (7, 5):
        if opcode.startswith('CS2R'):
            return [dst] + src, [tmp_dst] + tmp_src
        else:
            return src, [tmp_dst]
    elif cc == (7, 0):
        if opcode.startswith('CS2R'):
            return [dst] + src, [tmp_dst] + tmp_src
        else:
            return src, [tmp_dst]
    elif cc == (8, 0):
        return src, [tmp_dst]
    elif cc == (8, 6):
        return src, [tmp_dst]
    else:
        raise RuntimeError(f'unsupported compute capability: {cc}')


def get_st_window(cc):
    if cc == (7, 5):
        return 10
    elif cc == (7, 0):
        return 10
    elif cc == (8, 0):
        return 8
    elif cc == (8, 6):
        return 8
    else:
        raise RuntimeError(f'unsupported compute capability: {cc}')


def check_ban_opcode(cc, opcode):
    if cc == (7, 5):
        if opcode.startswith('ERRBAR'):
            return False
        elif opcode.startswith('MEMBAR'):
            return False
        elif opcode.startswith('EXIT'):
            return False
        elif opcode.startswith('BAR'):
            return False
        elif opcode.startswith('DEPBAR'):
            return False
        elif opcode.startswith('ULDC'):
            return False
        return True
    elif cc == (7, 0):
        if opcode.startswith('ERRBAR'):
            return False
        elif opcode.startswith('MEMBAR'):
            return False
        elif opcode.startswith('EXIT'):
            return False
        elif opcode.startswith('BAR'):
            return False
        elif opcode.startswith('DEPBAR'):
            return False
        elif opcode.startswith('ULDC'):
            return False
        return True
    elif cc == (8, 0):
        if opcode.startswith('LDGDEPBAR'):
            return False
        elif opcode.startswith('DEPBAR'):
            return False
        elif opcode.startswith('EXIT'):
            return False
        elif opcode.startswith('BAR.SYNC'):
            return False
        return True
    elif cc == (8, 6):
        if opcode.startswith('LDGDEPBAR'):
            return False
        elif opcode.startswith('DEPBAR'):
            return False
        elif opcode.startswith('EXIT'):
            return False
        elif opcode.startswith('BAR.SYNC'):
            return False
        return True
    else:
        raise RuntimeError(f'unsupported compute capability: {cc}')


def check_adj_opcodes(cc, prev_opcode, cur_opcode, prev_dst, cur_dst):
    if cc == (7, 5):
        return True
    elif cc == (7, 0):
        return True
    elif cc == (8, 0):
        if prev_opcode.startswith('LDGSTS') and cur_opcode.startswith(
                'LDGSTS'):
            if prev_dst == cur_dst:
                # it seems LDGSTS follows certain order
                return False
        elif prev_opcode.startswith('LDG') and cur_opcode.startswith('LOP3'):
            return False
        elif prev_opcode.startswith('STG') and cur_opcode.startswith('STG'):
            return False
        return True
    elif cc == (8, 6):
        if prev_opcode.startswith('LDGSTS') and cur_opcode.startswith(
                'LDGSTS'):
            if prev_dst == cur_dst:
                # it seems LDGSTS follows certain order
                return False
        elif prev_opcode.startswith('LDG') and cur_opcode.startswith('LOP3'):
            return False
        elif prev_opcode.startswith('STG') and cur_opcode.startswith('STG'):
            return False
        return True
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
