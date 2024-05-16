import os
import pickle
import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch

import triton
import triton.language as tl

import random
import numpy as np

from cuasmrl.jit import jit
from cuasmrl.autotuner import autotune as fgk_autotune
from cuasmrl.utils.gpu_utils import get_gpu_name, get_gpu_cc

from cuasmrl.autotuner import triton_autotune_with_cache
from cuasmrl.bench import do_bench

# yapf: disable
@dataclass
class Config:
    # Kernel
    default_out_path: str = "data"
    seed: int = 1337
    n_tests: int = 2
    load: Optional[str] = None
    bench: int = 0
    tt: bool = False

    # Workload
    b: int = 1
    m: int = 4
    n: int = 64
    k: int = 64

    # RL
    train: int = 1
    log: int = 1
    verbose: int = 0
    ## Env
    env_id: str = 'cuasmenv-v0'
    num_env: int = 1
    num_iterations: int = int(1e3)
    minibatch_size: int = 8
    horizon: int = 32
    num_steps: int = 64
    normalize_reward: int = 0
    ckpt_freq: int = 100
    ## Agent
    agent: str = "ppo"
    weights_path: Optional[str] = None
    agent_id: Optional[str] = None
    anneal_lr: int = 1
    gae: int = 1
    norm_adv: int = 1
    clip_vloss: int = 1
    update_epochs: int = 4
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    gpu: int = 0


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="???")

    # Add arguments to the parser
    parser.add_argument("--default_out_path", type=str, default="data")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n_tests", type=int, default=2)
    parser.add_argument("--load", type=str)
    parser.add_argument("--bench", type=int, default=0)
    parser.add_argument('--tt', default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("-b", type=int,  default=2)
    parser.add_argument("-m", type=int,  default=512)
    parser.add_argument("-n", type=int, default=512)
    parser.add_argument("-k", type=int,  default=2048)

    parser.add_argument("-t", "--train", type=int, dest="train", default=1)
    parser.add_argument("-l", "--log", type=int, dest="log", default=1)
    parser.add_argument("--verbose", type=int, default=0)

    parser.add_argument("--env_id", type=str, default='cuasmenv-v0')
    parser.add_argument("--num_iterations", type=int, default=int(200))
    parser.add_argument("--minibatch_size", type=int, default=8)
    parser.add_argument("--horizon", type=int, dest="horizon", default=32)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--normalize_reward", type=int, default=0)
    parser.add_argument("--ckpt_freq", type=int, default=100)

    parser.add_argument("--agent", type=str, default="ppo")
    parser.add_argument("--weights_path", type=str)
    parser.add_argument("--agent_id", type=str)
    parser.add_argument("--anneal_lr", type=int, default=1)
    parser.add_argument("--gae", type=int, default=1)
    parser.add_argument("--norm_adv", type=int, default=1)
    parser.add_argument("--clip_vloss", type=int, default=1)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    config = Config(**vars(args))
    return config


GPU = get_gpu_name()

# CREDITS: Initially inspired by the Triton tutorial


def call_tt(kernel, a, b):
    # checks constraints
    assert a.shape[2] == b.shape[1], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    batch_size, M, K = a.shape
    _, K, N = b.shape
    # assert (
    #         K % 32 == 0
    # ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_K_SIZE"
    # allocates output
    c = torch.empty((batch_size, M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_M_SIZE"]) * triton.cdiv(N, META["BLOCK_N_SIZE"]),
        batch_size,
    )
    kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), a.stride(2), b.stride(0), b.stride(1), b.stride(2), c.stride(0), c.stride(1), c.stride(2),
    )
    return c

def call(kernel, load_dir, a, b):
    # checks constraints
    assert a.shape[2] == b.shape[1], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    batch_size, M, K = a.shape
    _, K, N = b.shape
    # assert (
    #         K % 32 == 0
    # ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_K_SIZE"
    # allocates output
    c = torch.empty((batch_size, M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_M_SIZE"]) * triton.cdiv(N, META["BLOCK_N_SIZE"]),
        batch_size,
    )
    kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), a.stride(2), b.stride(0), b.stride(1), b.stride(2), c.stride(0), c.stride(1), c.stride(2),
        load_dir=load_dir,
    )
    return c


if __name__ == '__main__':
    drl_config = parse_args()

    random.seed(drl_config.seed)
    np.random.seed(drl_config.seed)
    torch.manual_seed(drl_config.seed)

    B, M, N, K = drl_config.b, drl_config.m, drl_config.n, drl_config.k

    a = torch.randn((B, M, K), device="cuda", dtype=torch.float16, requires_grad=False)
    b = torch.randn((B, K, N), device="cuda", dtype=torch.float16, requires_grad=False)


    drl_config.total_flops = 2*B * M * N * K
    drl_config.save_dir = f'{GPU}/bmm/{B}_{M}_{N}_{K}'

    if drl_config.load is None:
        load_dir = None
    elif drl_config.load == "auto":
        load_dir = f'data/{GPU}/bmm/{B}_{M}_{N}_{K}'
    else:
        load_dir = drl_config.load

    @fgk_autotune(
        configs=[
            triton.Config({'BLOCK_M_SIZE': 128, 'BLOCK_N_SIZE': 256, 'BLOCK_K_SIZE': 64, 'GROUP_M_SIZE': 8}, num_stages=1,
                          num_warps=8),
            triton.Config({'BLOCK_M_SIZE': 64, 'BLOCK_N_SIZE': 256, 'BLOCK_K_SIZE': 32, 'GROUP_M_SIZE': 8}, num_stages=4,
                          num_warps=4),
            triton.Config({'BLOCK_M_SIZE': 128, 'BLOCK_N_SIZE': 128, 'BLOCK_K_SIZE': 32, 'GROUP_M_SIZE': 8}, num_stages=4,
                          num_warps=4),
            triton.Config({'BLOCK_M_SIZE': 64, 'BLOCK_N_SIZE': 32, 'BLOCK_K_SIZE': 32, 'GROUP_M_SIZE': 8}, num_stages=5,
                          num_warps=2),
            triton.Config({'BLOCK_M_SIZE': 64, 'BLOCK_N_SIZE': 32, 'BLOCK_K_SIZE': 32, 'GROUP_M_SIZE': 8}, num_stages=2,
                        num_warps=2),
        ],
        key=['m_size'],
        ret_ptr=2,
        drl_config=drl_config,
    )
    @jit
    def cuasmrl(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        m_size, n_size, k_size,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
        # by to get the element one row down (A has M rows)
        a_batch_stride, a_m_stride, a_k_stride, b_batch_stride, b_k_stride, b_n_stride, c_batch_stride, c_m_stride, c_n_stride,
        # Meta-parameters
        BLOCK_M_SIZE: tl.constexpr,
        BLOCK_N_SIZE: tl.constexpr,
        BLOCK_K_SIZE: tl.constexpr,
        GROUP_M_SIZE: tl.constexpr,
    ):
        """Kernel for computing the matmul C = A x B.
        A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        """
        # -----------------------------------------------------------
        # To see later
        batch_idx = tl.program_id(axis=1)
        # program ID
        program_idx = tl.program_id(axis=0)

        # number of program ids along the M axis
        program_m_count = tl.cdiv(m_size, BLOCK_M_SIZE)
        # number of programs ids along the N axis
        program_n_count = tl.cdiv(n_size, BLOCK_N_SIZE)

        # number of programs in group
        program_in_group_count = GROUP_M_SIZE * program_n_count
        # id of the group this program is in
        group_idx = program_idx // program_in_group_count
        # row-id of the first program in the group
        first_program_m_idx = group_idx * GROUP_M_SIZE
        # if `program_m_count` isn't divisible by `GROUP_M_SIZE`, the last group is smaller
        GROUP_M_SIZE = min(program_m_count - first_program_m_idx, GROUP_M_SIZE)
        # *within groups*, programs are ordered in a column-major order
        # row-id of the program in the *launch grid*
        program_m_idx = first_program_m_idx + (program_idx % GROUP_M_SIZE)
        # col-id of the program in the *launch grid*
        program_n_idx = (program_idx % program_in_group_count) // GROUP_M_SIZE

        # ----------------------------------------------------------
        a_offs = program_m_idx * BLOCK_M_SIZE + tl.arange(0, BLOCK_M_SIZE)
        b_offs = program_n_idx * BLOCK_N_SIZE + tl.arange(0, BLOCK_N_SIZE)

        k_range_offs = tl.arange(0, BLOCK_K_SIZE)


        a_ptrs = a_ptr + a_batch_stride * batch_idx + (a_offs[:, None] * a_m_stride + k_range_offs[None, :] * a_k_stride)
        b_ptrs = b_ptr + b_batch_stride * batch_idx + (k_range_offs[:, None] * b_k_stride + b_offs[None, :] * b_n_stride)

        # -----------------------------------------------------------
        accumulator = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)
        for k in range(0, k_size, BLOCK_K_SIZE):

            a_ptr_mask = (a_offs[:, None] < m_size) & (k_range_offs[None, :] < k_size)
            a = tl.load(a_ptrs, mask=a_ptr_mask, other=0)

            b_ptr_mask = (k_range_offs[:, None] < k_size) & (b_offs[None, :] < n_size)
            b = tl.load(b_ptrs, mask=b_ptr_mask, other=0)

            # We accumulate along the K dimension
            accumulator += tl.dot(a, b)
            # Advance the ptrs to the next K block
            a_ptrs += BLOCK_K_SIZE * a_k_stride
            b_ptrs += BLOCK_K_SIZE * b_k_stride

        c = accumulator.to(tl.float16)

        # -----------------------------------------------------------
        c_m_offs = program_m_idx * BLOCK_M_SIZE + tl.arange(0, BLOCK_M_SIZE)
        c_n_offs = program_n_idx * BLOCK_N_SIZE + tl.arange(0, BLOCK_N_SIZE)
        c_ptrs = c_ptr + c_batch_stride * batch_idx + c_m_stride * c_m_offs[:, None] + c_n_stride * c_n_offs[None, :]
        c_ptr_mask = (c_m_offs[:, None] < m_size) & (c_n_offs[None, :] < n_size)
        tl.store(c_ptrs, c, mask=c_ptr_mask)

    @triton_autotune_with_cache(
        configs=[
            triton.Config({'BLOCK_M_SIZE': 128, 'BLOCK_N_SIZE': 256, 'BLOCK_K_SIZE': 64, 'GROUP_M_SIZE': 8}, num_stages=1,
                          num_warps=8),
            triton.Config({'BLOCK_M_SIZE': 64, 'BLOCK_N_SIZE': 256, 'BLOCK_K_SIZE': 32, 'GROUP_M_SIZE': 8}, num_stages=4,
                          num_warps=4),
            triton.Config({'BLOCK_M_SIZE': 128, 'BLOCK_N_SIZE': 128, 'BLOCK_K_SIZE': 32, 'GROUP_M_SIZE': 8}, num_stages=4,
                          num_warps=4),
            triton.Config({'BLOCK_M_SIZE': 64, 'BLOCK_N_SIZE': 32, 'BLOCK_K_SIZE': 32, 'GROUP_M_SIZE': 8}, num_stages=5,
                          num_warps=2),
            triton.Config({'BLOCK_M_SIZE': 64, 'BLOCK_N_SIZE': 32, 'BLOCK_K_SIZE': 32, 'GROUP_M_SIZE': 8}, num_stages=2,
                        num_warps=2),
        ],
        key=['m_size'],
        drl_config=drl_config,
    )
    @triton.jit
    def tt(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        # Matrix dimensions
        m_size,
        n_size,
        k_size,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
        # by to get the element one row down (A has M rows)
        a_batch_stride,
        a_m_stride,
        a_k_stride,
        b_batch_stride,
        b_k_stride,
        b_n_stride,
        c_batch_stride,
        c_m_stride,
        c_n_stride,
        # Meta-parameters
        BLOCK_M_SIZE: tl.constexpr,
        BLOCK_N_SIZE: tl.constexpr,
        BLOCK_K_SIZE: tl.constexpr,
        GROUP_M_SIZE: tl.constexpr,
    ):
        """Kernel for computing the matmul C = A x B.
        A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        """
        # -----------------------------------------------------------
        # Map program ids `program_idx` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse
        # See above `L2 Cache Optimizations` section for details

        # Supergrouping of blocks
        # To see later
        batch_idx = tl.program_id(axis=1)
        # program ID
        program_idx = tl.program_id(axis=0)

        # number of program ids along the M axis
        program_m_count = tl.cdiv(m_size, BLOCK_M_SIZE)
        # number of programs ids along the N axis
        program_n_count = tl.cdiv(n_size, BLOCK_N_SIZE)

        # number of programs in group
        program_in_group_count = GROUP_M_SIZE * program_n_count
        # id of the group this program is in
        group_idx = program_idx // program_in_group_count
        # row-id of the first program in the group
        first_program_m_idx = group_idx * GROUP_M_SIZE
        # if `program_m_count` isn't divisible by `GROUP_M_SIZE`, the last group is smaller
        GROUP_M_SIZE = min(program_m_count - first_program_m_idx, GROUP_M_SIZE)
        # *within groups*, programs are ordered in a column-major order
        # row-id of the program in the *launch grid*
        program_m_idx = first_program_m_idx + (program_idx % GROUP_M_SIZE)
        # col-id of the program in the *launch grid*
        program_n_idx = (program_idx % program_in_group_count) // GROUP_M_SIZE

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # a_ptrs is a block of [BLOCK_M_SIZE, BLOCK_K_SIZE] pointers
        # b_ptrs is a block of [BLOCK_K_SIZE, BLOCK_N_SIZE] pointers
        # see above `Pointer Arithmetics` section for details

        # program_m_idx * BLOCK_M_SIZE is the row index of the first element of the block of size BLOCK_M_SIZE
        # We add tl.arange(0, BLOCK_M_SIZE) to get a vector of row indexes
        a_offs = program_m_idx * BLOCK_M_SIZE + tl.arange(0, BLOCK_M_SIZE)
        b_offs = program_n_idx * BLOCK_N_SIZE + tl.arange(0, BLOCK_N_SIZE)

        k_range_offs = tl.arange(0, BLOCK_K_SIZE)

        # a_offs[:, None] is a column vector of BLOCK_M_SIZE rows indexes
        # We multiply by stride_am, to we get a column vector of memory offsets to each start of a row
        # k_range_offs[None, :] is a row vector of size BLOCK_K_SIZE columns indexes
        # We multiply stride_ak to get a row vector of memory offsets to each start of a column
        # When we add both. We get a matrix of memory offsets.
        # For A in RowMajor stride_ak will be 1, so k_range_offs[None, :] * stride_ak will be
        # just 0,1,2,3,4,5....BLOCK_K_SIZE
        a_ptrs = a_ptr + a_batch_stride * batch_idx + (a_offs[:, None] * a_m_stride + k_range_offs[None, :] * a_k_stride)
        b_ptrs = b_ptr + b_batch_stride * batch_idx + (k_range_offs[:, None] * b_k_stride + b_offs[None, :] * b_n_stride)

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix
        # We accumulate into a `[BLOCK_M_SIZE, BLOCK_N_SIZE]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop
        accumulator = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)
        for k in range(0, k_size, BLOCK_K_SIZE):
            # Note that for simplicity, we don't apply a mask here.
            # This means that if K is not a multiple of BLOCK_K_SIZE,
            # this will access out-of-bounds memory and produce an
            # error or (worse!) incorrect results.

            a_ptr_mask = (a_offs[:, None] < m_size) & (k_range_offs[None, :] < k_size)
            a = tl.load(a_ptrs, mask=a_ptr_mask, other=0)

            b_ptr_mask = (k_range_offs[:, None] < k_size) & (b_offs[None, :] < n_size)
            b = tl.load(b_ptrs, mask=b_ptr_mask, other=0)

            # We accumulate along the K dimension
            accumulator += tl.dot(a, b)
            # Advance the ptrs to the next K block
            a_ptrs += BLOCK_K_SIZE * a_k_stride
            b_ptrs += BLOCK_K_SIZE * b_k_stride

        c = accumulator.to(tl.float16)

        # -----------------------------------------------------------
        # Write back the block of the output matrix C
        c_m_offs = program_m_idx * BLOCK_M_SIZE + tl.arange(0, BLOCK_M_SIZE)
        c_n_offs = program_n_idx * BLOCK_N_SIZE + tl.arange(0, BLOCK_N_SIZE)
        c_ptrs = c_ptr + c_batch_stride * batch_idx + c_m_stride * c_m_offs[:, None] + c_n_stride * c_n_offs[None, :]
        c_ptr_mask = (c_m_offs[:, None] < m_size) & (c_n_offs[None, :] < n_size)
        tl.store(c_ptrs, c, mask=c_ptr_mask)

    call(cuasmrl, load_dir, a, b)

    if drl_config.tt:
        out_rms_triton = call_tt(tt, a, b)