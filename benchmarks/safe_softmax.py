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
    m: int = 1024
    n: int =16384
    bm: int = 8
    bn: int = 64

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

    parser.add_argument("-m", type=int, default=512)
    parser.add_argument("-n", type=int, default=32000)
    parser.add_argument("--bm", type=int, default=16)
    parser.add_argument("--bn", type=int, default=128)

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



@triton.jit
def tt_softmax(
    x_ptr,
    y_ptr,
    x_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    m_offset = pid_m * x_stride * BLOCK_M
    k_offset = tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * x_stride + k_offset[None, :])
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    for k in range(0, tl.cdiv(x_stride, BLOCK_N)):
        mask = k * BLOCK_N + k_offset < x_stride
        x = tl.load(x_ptrs, mask[None, :], -float('inf'))
        m_ij = tl.max(x, 1)
        m_i = tl.maximum(m_ij, m_i)
        x_ptrs += BLOCK_N

    x_ptrs = x_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * x_stride + k_offset[None, :])
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k in range(0, tl.cdiv(x_stride, BLOCK_N)):
        mask = k * BLOCK_N + k_offset < x_stride
        x = tl.load(x_ptrs, mask[None, :], 0)
        l_ij = tl.exp(x - m_i[:, None])
        l_i += tl.sum(l_ij, 1)
        x_ptrs += BLOCK_N

    x_ptrs = x_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * x_stride + k_offset[None, :])
    y_ptrs = y_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * x_stride + k_offset[None, :])
    for k in range(0, tl.cdiv(x_stride, BLOCK_N)):
        mask = k * BLOCK_N + k_offset < x_stride
        x = tl.load(x_ptrs, mask[None, :], 0)

        numerator = tl.exp(x - m_i[:, None])
        denominator = l_i[:, None]
        y = numerator / denominator

        tl.store(y_ptrs, y, mask[None, :])

        x_ptrs += BLOCK_N
        y_ptrs += BLOCK_N


def call_tt(x, kernel, BLOCK_M, BLOCK_N):
    grid = (triton.cdiv(x.shape[0], BLOCK_M),
        )
    out = torch.empty_like(x)

    kernel[grid](
        x,
        out,

        x.stride(0),

        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out


def call(x, kernel, load_dir, BLOCK_M, BLOCK_N):
    grid = (triton.cdiv(x.shape[0], BLOCK_M),
        )
    out = torch.empty_like(x)

    kernel[grid](
        x,
        out,

        x.stride(0),

        # BLOCK_M=BLOCK_M,
        # BLOCK_N=BLOCK_N,

        load_dir=load_dir,
    )
    return out


if __name__ == "__main__":
    drl_config = parse_args()

    random.seed(drl_config.seed)
    np.random.seed(drl_config.seed)
    torch.manual_seed(drl_config.seed)

    device = torch.device("cuda:0")

    m, n = drl_config.m , drl_config.n
    BLOCK_M, BLOCK_N = drl_config.bm, drl_config.bn
    a = torch.randn((m, n), dtype=torch.float16, device=device)

    drl_config.total_flops = 2 * a.nelement() * a.element_size()
    drl_config.save_dir = f'{GPU}/safe_softmax/{m}_{n}_{BLOCK_M}_{BLOCK_N}'
    if drl_config.load is None:
        load_dir = None
    elif drl_config.load == "auto":
        load_dir = f'data/{GPU}/safe_softmax/{m}_{n}_{BLOCK_M}_{BLOCK_N}'
    else:
        load_dir = drl_config.load

    @fgk_autotune(
        configs=[
            triton.Config({'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N}, num_stages=4, num_warps=4),
        ],
        key=['BLOCK_M', 'BLOCK_N'],
        drl_config=drl_config,  # just need the path really
        ret_ptr=1,
    )
    @jit
    def _cuasmrl(
        x_ptr,
        y_ptr,
        x_stride,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)

        m_offset = pid_m * x_stride * BLOCK_M
        k_offset = tl.arange(0, BLOCK_N)

        x_ptrs = x_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * x_stride + k_offset[None, :])
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        for k in range(0, tl.cdiv(x_stride, BLOCK_N)):
            mask = k * BLOCK_N + k_offset < x_stride
            x = tl.load(x_ptrs, mask[None, :], -float('inf'))
            m_ij = tl.max(x, 1)
            m_i = tl.maximum(m_ij, m_i)
            x_ptrs += BLOCK_N

        x_ptrs = x_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * x_stride + k_offset[None, :])
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        for k in range(0, tl.cdiv(x_stride, BLOCK_N)):
            mask = k * BLOCK_N + k_offset < x_stride
            x = tl.load(x_ptrs, mask[None, :], 0)
            l_ij = tl.exp(x - m_i[:, None])
            l_i += tl.sum(l_ij, 1)
            x_ptrs += BLOCK_N

        x_ptrs = x_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * x_stride + k_offset[None, :])
        y_ptrs = y_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * x_stride + k_offset[None, :])
        for k in range(0, tl.cdiv(x_stride, BLOCK_N)):
            mask = k * BLOCK_N + k_offset < x_stride
            x = tl.load(x_ptrs, mask[None, :], 0)

            numerator = tl.exp(x - m_i[:, None])
            denominator = l_i[:, None]
            y = numerator / denominator

            tl.store(y_ptrs, y, mask[None, :])

            x_ptrs += BLOCK_N
            y_ptrs += BLOCK_N



    call(a, _cuasmrl, load_dir, BLOCK_M, BLOCK_N)

    if drl_config.tt:
        triton_out = call_tt(a, tt_softmax, BLOCK_M, BLOCK_N)

    if bool(drl_config.bench):
        print('BENCH...')
        torch.cuda.synchronize()

        ms = do_bench(lambda: call(a, _cuasmrl, load_dir, BLOCK_M, BLOCK_N), warmup=100, rep=100)
        ms_tt = do_bench(lambda: call_tt(a, tt_softmax, BLOCK_M, BLOCK_N), warmup=100, rep=100)

        data = {
            'cuasmrl': ms,
            'tt': ms_tt,
        }

        print(data)

        fp = f"data/{GPU}/safe_softmax/{m}_{n}_{BLOCK_M}_{BLOCK_N}/bench_{drl_config.seed}.pkl"
        with open(fp, 'wb') as f:
            pickle.dump(data, f)
