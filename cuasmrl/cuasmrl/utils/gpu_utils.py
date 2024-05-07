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
    # RTX3000
    (8, 6): (
        # ['LDGSTS', 'LDG', 'STG'],
        # ['LDSM', 'LDS', 'LDGSTS', 'LDG', 'STG'],
        # ['LDGDEPBAR', 'DEPBAR', 'EXIT', 'BAR.SYNC', 'BRA'],  # ban_ops
        ['LDS', 'LDGSTS', 'LDG', 'STG'],
        ['LDSM', 'LDGDEPBAR', 'DEPBAR', 'EXIT', 'BAR.SYNC', 'BRA'],  # ban_ops
    ),
    # A100
    (8, 0): (
        # ['LDGSTS', 'LDG', 'STG'],
        # ['LDSM', 'LDS', 'LDGSTS', 'LDG', 'STG'],
        # ['LDGDEPBAR', 'DEPBAR', 'EXIT', 'BAR.SYNC', 'BRA'],  # ban_ops
        ['LDS', 'LDGSTS', 'LDG', 'STG'],
        ['LDSM', 'LDGDEPBAR', 'DEPBAR', 'EXIT', 'BAR.SYNC', 'BRA'],  # ban_ops
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
    # V100
    (7, 0): (
        # memory_ops
        [
            'LDG',
            # 'LDS',
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


def get_mutatable_ops(cc):
    if cc in MUTATABLE_OPS:
        return MUTATABLE_OPS[cc]
    else:
        raise RuntimeError(f'unsupported compute capability: {cc}')


def has_hazard(cc, st, opcode, dst, src, tmp_opcode, tmp_dst, tmp_src):
    if cc == (7, 0):
        # st
        min_st = 8
        if tmp_opcode.startswith('LDG'):
            min_st = 15
        elif opcode.startswith('LDG'):
            min_st = 15

        # hazard
        if st <= min_st:
            # wAw
            if dst == tmp_dst:
                return True
            # RAW; WAR
            if dst in tmp_src:
                return True
            if tmp_dst in src:
                return True

        return False

    elif cc == (8, 0):
        # st
        min_st = 12
        # from rbe
        # elif opcode.startswith('LDSM'):
        #     min_st = 11
        # elif tmp_opcode.startswith('IADD3'):
        #     min_st = 10
        # flash-decoding
        # if opcode.startswith('LDG') and tmp_opcode.startswith('IADD3'):
        #     min_st = 12
        # if opcode.startswith('LDG') and tmp_opcode.startswith('PRMT'):
        #     min_st = 6

        # hazard
        if st <= min_st:
            # write-after-write
            if dst == tmp_dst:
                return True
            # RAW; WAR
            if dst in tmp_src:
                return True
            if tmp_dst in src:
                return True

        return False
    else:
        raise RuntimeError(f'unsupported compute capability: {cc}')


def get_min_stall_count(cc, opcode, tmp_opcode):
    if cc == (7, 5):
        if opcode.startswith('CS2R'):
            return 19
        return 7
    elif cc == (7, 0):
        return 12
    elif cc == (8, 0):
        return 12
    elif cc == (8, 6):
        return 12
    else:
        raise RuntimeError(f'unsupported compute capability: {cc}')


def get_st_window(cc):
    if cc == (7, 5):
        return 10
    elif cc == (7, 0):
        return 10
    elif cc == (8, 0):
        return 10
    elif cc == (8, 6):
        return 10
    else:
        raise RuntimeError(f'unsupported compute capability: {cc}')


def check_adj_opcodes(
    cc,
    prev_opcode,
    cur_opcode,
    prev_dst,
    cur_dst,
    prev_src,
    cur_src,
    prev_predicate,
    cur_predicate,
):
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
        if prev_opcode.startswith('LDG') and cur_opcode.startswith('LOP3'):
            return False
        if prev_opcode.startswith('STG') and cur_opcode.startswith('STG'):
            return False

        # LDGEPBAR
        if prev_opcode.startswith('LDGDEPBAR') and cur_opcode.startswith(
                'DEPBAR'):
            return False
        # if prev_opcode.startswith('LDGDEPBAR') and cur_opcode.startswith('LDGSTS'):
        #     return False
        if prev_opcode.startswith('LDGSTS') and cur_opcode.startswith(
                'LDGDEPBAR'):
            return False
        # if prev_opcode.startswith('DEPBAR') and cur_opcode.startswith('LDGDEPBAR'):
        #     return False

        # from bmm
        if prev_opcode.startswith('MOV') and cur_opcode.startswith('LDS.128'):
            return False

        # from ff
        if cur_opcode.startswith('LDGSTS'):
            if prev_dst == cur_dst:
                return False
        if prev_opcode.startswith('LDGSTS') and cur_opcode.startswith(
                'UIADD3.X'):
            return False

        # LDS/LDSM must load from consecutive memory address...
        if prev_opcode.startswith('LDS') and cur_opcode.startswith('LDS'):
            if set(prev_src).intersection(cur_src):
                return False

        # from conv
        # if prev_opcode.startswith('LDG') and cur_opcode.startswith('CS2R'):
        #     return False
        # from int4
        # elif prev_opcode.startswith('LDG') and cur_opcode.startswith('IMAD.X'):
        #     return False

        # predicate dependencies
        if cur_predicate is not None:
            if cur_predicate.startswith('@'):
                cur_predicate = cur_predicate[1:]
            if cur_predicate == prev_dst:
                return False
        if prev_predicate is not None:
            if prev_predicate.startswith('@'):
                prev_predicate = prev_predicate[1:]
            if prev_predicate == cur_dst:
                return False

        return True
    elif cc == (8, 6):
        if prev_opcode.startswith('LDGSTS') and cur_opcode.startswith(
                'LDGSTS'):
            if prev_dst == cur_dst:
                # it seems LDGSTS follows certain order
                return False
        if prev_opcode.startswith('LDG') and cur_opcode.startswith('LOP3'):
            return False
        if prev_opcode.startswith('STG') and cur_opcode.startswith('STG'):
            return False

        # LDGEPBAR
        if prev_opcode.startswith('LDGDEPBAR') and cur_opcode.startswith(
                'DEPBAR'):
            return False
        # if prev_opcode.startswith('LDGDEPBAR') and cur_opcode.startswith('LDGSTS'):
        #     return False
        if prev_opcode.startswith('LDGSTS') and cur_opcode.startswith(
                'LDGDEPBAR'):
            return False
        # if prev_opcode.startswith('DEPBAR') and cur_opcode.startswith('LDGDEPBAR'):
        #     return False

        # from bmm
        if prev_opcode.startswith('MOV') and cur_opcode.startswith('LDS.128'):
            return False
        if prev_opcode.startswith('LDS') and cur_opcode.startswith('LDS'):
            if set(prev_src).intersection(cur_src):
                return False

        # LDSM must load from consecutive memory address...

        # from conv
        # if prev_opcode.startswith('LDG') and cur_opcode.startswith('CS2R'):
        #     return False
        # from int4
        # elif prev_opcode.startswith('LDG') and cur_opcode.startswith('IMAD.X'):
        #     return False

        # predicate dependencies
        if cur_predicate is not None:
            if cur_predicate.startswith('@'):
                cur_predicate = cur_predicate[1:]
            if cur_predicate == prev_dst:
                return False
        if prev_predicate is not None:
            if prev_predicate.startswith('@'):
                prev_predicate = prev_predicate[1:]
            if prev_predicate == cur_dst:
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
