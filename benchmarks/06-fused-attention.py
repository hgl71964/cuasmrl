import os
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

# yapf: disable
@dataclass
class Config:
    # Kernel
    default_out_path: str = "data"
    seed: int = 1337
    n_tests: int = 10
    load: Optional[str] = None
    bench: int = 0

    # Workload
    Z: int = 1
    H: int = 4
    wl: int = 16384
    D_HEAD: int = 64

    # RL
    train: int = 1
    log: int = 1
    verbose: int = 0
    ## Env
    env_id: str = 'cuasmenv-v0'
    num_env: int = 1
    num_iterations: int = int(1e4)
    minibatch_size: int = 8
    horizon: int = 32
    num_steps: int = 64
    normalize_reward: int = 0
    ckpt_freq: int = 10
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
    parser.add_argument("--n_tests", type=int, default=10)
    parser.add_argument("--load", type=str)
    parser.add_argument("--bench", type=int, default=0)

    parser.add_argument("--Z", type=int, dest="Z", default=1)
    parser.add_argument("--H", type=int, dest="H", default=4)
    parser.add_argument("--wl", type=int, default=16384)
    parser.add_argument("--dh", type=int, dest="D_HEAD", default=64)

    parser.add_argument("-t", "--train", type=int, dest="train", default=1)
    parser.add_argument("-l", "--log", type=int, dest="log", default=1)
    parser.add_argument("--verbose", type=int, default=0)

    parser.add_argument("--env_id", type=str, default='cuasmenv-v0')
    parser.add_argument("--num_iterations", type=int, default=int(1e4))
    parser.add_argument("--minibatch_size", type=int, default=8)
    parser.add_argument("--horizon", type=int, dest="horizon", default=32)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--normalize_reward", type=int, default=0)
    parser.add_argument("--ckpt_freq", type=int, default=10)

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
def _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,  #
        K_block_ptr,
        V_block_ptr,  #
        start_m,
        qk_scale,  #
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,  #
        STAGE: tl.constexpr,
        offs_m: tl.constexpr,
        offs_n: tl.constexpr,  #
        N_CTX: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        acc += tl.dot(p.to(tl.float16), v)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.autotune(
    configs=[
        triton.Config({ 'BLOCK_M': 128, 'BLOCK_N': 64 }, num_stages=2, num_warps=4),
        triton.Config({ 'BLOCK_M': 128, 'BLOCK_N': 64 }, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
        triton.Config({ 'BLOCK_M': 128, 'BLOCK_N': 32 }, num_stages=3, num_warps=4),
        triton.Config({ 'BLOCK_M': 128, 'BLOCK_N': 32 }, num_stages=4, num_warps=4),
        triton.Config({ 'BLOCK_M': 128, 'BLOCK_N': 64 }, num_stages=3, num_warps=4),
        triton.Config({ 'BLOCK_M': 128, 'BLOCK_N': 64 }, num_stages=4, num_warps=4),
        triton.Config({ 'BLOCK_M': 128, 'BLOCK_N': 64 }, num_stages=3, num_warps=8),
        triton.Config({ 'BLOCK_M': 128, 'BLOCK_N': 64 }, num_stages=7, num_warps=8),
        triton.Config({ 'BLOCK_M': 128, 'BLOCK_N': 32 }, num_stages=7, num_warps=8),
        triton.Config({ 'BLOCK_M': 128, 'BLOCK_N': 32 }, num_stages=6, num_warps=8),
        triton.Config({ 'BLOCK_M': 128, 'BLOCK_N': 32 }, num_stages=5, num_warps=8),
        triton.Config({ 'BLOCK_M': 128, 'BLOCK_N': 32 }, num_stages=4, num_warps=8),
        triton.Config({ 'BLOCK_M': 128, 'BLOCK_N': 64 }, num_stages=6, num_warps=4),
    ],
    key=['N_CTX'],
)
@triton.jit
def _attn_fwd_triton(
        Q, K, V, sm_scale, M, Out,  #
        stride_qz, stride_qh, stride_qm, stride_qk,  #
        stride_kz, stride_kh, stride_kn, stride_kk,  #
        stride_vz, stride_vh, stride_vk, stride_vn,  #
        stride_oz, stride_oh, stride_om, stride_on,  #
        Z, H,  #
        N_CTX: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_DMODEL: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        STAGE: tl.constexpr  #
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(
        tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX  #
        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        tl.debug_barrier()
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,  #
            2,
            offs_m,
            offs_n,
            N_CTX  #
        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def attn_forward(q, k, v, M, o, grid, causal, sm_scale, kernel, load_dir):
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    stage = 3 if causal else 1

    # q, k, v: (Z, H, N_CTX, D_HEAD)
    kernel[grid](
        q, k, v, sm_scale, M, o,  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
        q.shape[0], q.shape[1],  #
        N_CTX=q.shape[2],  #
        # BLOCK_M=BLOCK_M,  #
        # BLOCK_N=BLOCK_N,  #
        BLOCK_DMODEL=Lk,  #
        STAGE=stage,  #
        # num_warps=num_warps,  #
        # num_stages=num_stages  #

        # gh512
        load_dir=load_dir,
    )
    return o


def triton_attn_forward(q, k, v, M, o, grid, causal, sm_scale, kernel):
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    stage = 3 if causal else 1

    # q, k, v: (Z, H, N_CTX, D_HEAD)
    kernel[grid](
        q, k, v, sm_scale, M, o,  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
        q.shape[0], q.shape[1],  #
        N_CTX=q.shape[2],  #
        # BLOCK_M=BLOCK_M,  #
        # BLOCK_N=BLOCK_N,  #
        BLOCK_DMODEL=Lk,  #
        STAGE=stage,  #
        # num_warps=num_warps,  #
        # num_stages=num_stages  #
    )
    return o


def main():

    config = parse_args()

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True

    # workload
    Z, H, N_CTX, D_HEAD = config.Z, config.H, config.wl, config.D_HEAD
    dtype = torch.float16
    q = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype,
                     device="cuda").normal_(mean=0.0,
                                            std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype,
                     device="cuda").normal_(mean=0.0,
                                            std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype,
                     device="cuda").normal_(mean=0.0,
                                            std=0.5).requires_grad_())

    causal = True
    flops_per_matmul = 2.0 * Z * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    sm_scale = 0.5

    # args
    config.total_flops = total_flops
    config.save_dir = f'{GPU}/flash_attn/{Z}_{H}_{N_CTX}_{D_HEAD}'

    @fgk_autotune(
        configs=[
            triton.Config({ 'BLOCK_M': 128, 'BLOCK_N': 64 }, num_stages=2, num_warps=4),
            # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
            # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
            # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_stages=3, num_warps=8),
            # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
            # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
            # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
            # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
            # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
            # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
            # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=7, num_warps=8),
            # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=7, num_warps=8),
            # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=6, num_warps=8),
            # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=5, num_warps=8),
            # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=4, num_warps=8),
            # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=6, num_warps=4),
        ],
        key=['N_CTX'],
        ret_ptr=5,
        drl_config=config,
    )
    @jit
    def _attn_fwd(
            Q, K, V, sm_scale, M, Out,  #
            stride_qz, stride_qh, stride_qm, stride_qk,  #
            stride_kz, stride_kh, stride_kn, stride_kk,  #
            stride_vz, stride_vh, stride_vk, stride_vn,  #
            stride_oz, stride_oh, stride_om, stride_on,  #
            Z, H,  #
            N_CTX: tl.constexpr,  #
            BLOCK_M: tl.constexpr,  #
            BLOCK_DMODEL: tl.constexpr,  #
            BLOCK_N: tl.constexpr,  #
            STAGE: tl.constexpr  #
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(
            tl.int64) * stride_qh

        # block pointers
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(BLOCK_DMODEL, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(BLOCK_DMODEL, BLOCK_N),
            order=(0, 1),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        # initialize offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # load scales
        qk_scale = sm_scale
        qk_scale *= 1.44269504  # 1/log(2)
        # load q: it will stay in SRAM throughout
        q = tl.load(Q_block_ptr)
        # stage 1: off-band
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
        # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
        if STAGE & 1:
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,  #
                start_m,
                qk_scale,  #
                BLOCK_M,
                BLOCK_DMODEL,
                BLOCK_N,  #
                4 - STAGE,
                offs_m,
                offs_n,
                N_CTX  #
            )
        # stage 2: on-band
        if STAGE & 2:
            # barrier makes it easier for compielr to schedule the
            # two loops independently
            tl.debug_barrier()
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,  #
                start_m,
                qk_scale,  #
                BLOCK_M,
                BLOCK_DMODEL,
                BLOCK_N,  #
                2,
                offs_m,
                offs_n,
                N_CTX  #
            )
        # epilogue
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))

    # by default it is half
    o = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]),
                    device=q.device,
                    dtype=torch.float32)
    BLOCK_M = 128
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    if config.load is None:
        load_dir = None
    elif config.load == "auto":
        load_dir = f'data/{GPU}/flash_attn/{config.Z}_{config.H}_{config.wl}_{config.D_HEAD}'
    else:
        load_dir = config.load
    fgk_out = attn_forward(q, k, v, M, o, grid, causal, sm_scale, _attn_fwd,
                           load_dir)
    tri_out = triton_attn_forward(q, k, v, M, o, grid, causal, sm_scale,
                                  _attn_fwd_triton)

    ## TEST
    assert torch.allclose(tri_out, fgk_out, atol=1e-2, rtol=0)
    print('TEST PASSED')

    if not bool(config.bench):
        print('SKIP bench...')
        return

    torch.cuda.synchronize()
    try:
        from flash_attn.flash_attn_interface import \
            flash_attn_qkvpacked_func as flash_attn_func
        HAS_FLASH = True
    except BaseException:
        HAS_FLASH = False

    cc = get_gpu_cc()
    if cc[0] < 8:
        HAS_FLASH = False

    TORCH_HAS_FP8 = False

    print(f"use flash: {HAS_FLASH}; use fp8: {TORCH_HAS_FP8}")
    BATCH, N_HEADS, N_CTX, D_HEAD = config.Z, config.H, config.wl, config.D_HEAD

    configs = []
    # for mode in ["fwd", "bwd"]:
    for mode in ["fwd"]:
        # for causal in [True, False]:
        for causal in [True]:
            if mode == "bwd" and not causal:
                continue
            configs.append(
                triton.testing.Benchmark(
                    x_names=["N_CTX"],
                    # x_vals=[2**i for i in range(9, 14)],  # NOTE: e.g. if use 4096 cubin for 1024, it could fail
                    x_vals=[config.wl],
                    line_arg="provider",
                    line_vals=["fgk", "triton"] +
                    (["flash"] if HAS_FLASH else []),
                    line_names=["FGK", "Triton"] +
                    (["Flash-2"] if HAS_FLASH else []),
                    styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                    ylabel="ms",
                    plot_name=
                    f"fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}-causal={causal}",
                    args={
                        "H": N_HEADS,
                        "BATCH": BATCH,
                        "D_HEAD": D_HEAD,
                        "dtype": torch.float16,
                        "mode": mode,
                        "causal": causal,
                    },
                ))

    @triton.testing.perf_report(configs)
    def bench_flash_attention(BATCH,
                              H,
                              N_CTX,
                              D_HEAD,
                              causal,
                              mode,
                              provider,
                              dtype=torch.float16,
                              device="cuda"):
        print(f'[BENCH]: {provider};; {BATCH} {H} {N_CTX} {D_HEAD}')
        assert mode in ["fwd", "bwd"]
        warmup = 100
        rep = 100
        sm_scale = 0.5
        if provider == "fgk":
            q = torch.randn((BATCH, H, N_CTX, D_HEAD),
                            dtype=dtype,
                            device="cuda",
                            requires_grad=True)
            k = torch.randn((BATCH, H, N_CTX, D_HEAD),
                            dtype=dtype,
                            device="cuda",
                            requires_grad=True)
            if mode == "fwd" and TORCH_HAS_FP8:
                q = q.to(torch.float8_e5m2)
                k = k.to(torch.float8_e5m2)
            v = torch.randn((BATCH, H, N_CTX, D_HEAD),
                            dtype=dtype,
                            device="cuda",
                            requires_grad=True)
            o = torch.empty_like(q)
            M = torch.empty((q.shape[0], q.shape[1], q.shape[2]),
                            device=q.device,
                            dtype=torch.float32)
            grid = (triton.cdiv(q.shape[2],
                                BLOCK_M), q.shape[0] * q.shape[1], 1)
            if config.load is None:
                load_dir = None
            elif config.load == "auto":
                load_dir = f'data/{GPU}/flash_attn/{config.Z}_{config.H}_{config.wl}_{config.D_HEAD}'
            else:
                load_dir = config.load
            fn = lambda: attn_forward(q, k, v, M, o, grid, causal, sm_scale,
                                      _attn_fwd, load_dir)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        if provider == "triton":
            q = torch.randn((BATCH, H, N_CTX, D_HEAD),
                            dtype=dtype,
                            device="cuda",
                            requires_grad=True)
            k = torch.randn((BATCH, H, N_CTX, D_HEAD),
                            dtype=dtype,
                            device="cuda",
                            requires_grad=True)
            if mode == "fwd" and TORCH_HAS_FP8:
                q = q.to(torch.float8_e5m2)
                k = k.to(torch.float8_e5m2)
            v = torch.randn((BATCH, H, N_CTX, D_HEAD),
                            dtype=dtype,
                            device="cuda",
                            requires_grad=True)
            o = torch.empty_like(q)
            M = torch.empty((q.shape[0], q.shape[1], q.shape[2]),
                            device=q.device,
                            dtype=torch.float32)
            grid = (triton.cdiv(q.shape[2],
                                BLOCK_M), q.shape[0] * q.shape[1], 1)
            fn = lambda: triton_attn_forward(q, k, v, M, o, grid, causal,
                                             sm_scale, _attn_fwd_triton)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        if provider == "flash":
            qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD),
                              dtype=dtype,
                              device=device,
                              requires_grad=True)
            fn = lambda: flash_attn_func(qkv, causal=causal)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
        total_flops = 2 * flops_per_matmul
        if causal:
            total_flops *= 0.5
        if mode == "bwd":
            total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
        return total_flops / ms * 1e-9

    df = bench_flash_attention.run(print_data=True, return_df=True)
    if isinstance(df, list):
        assert len(df) == 1, f'expected 1 row, got {len(df)}'
        df = df[0]
    fp = f"data/{GPU}/results/flash_attn/{config.Z}_{config.H}_{config.wl}_{config.D_HEAD}_{config.seed}.pkl"
    if not os.path.exists(fp):
        if not os.path.exists(f"data/{GPU}/results/flash_attn"):
            os.makedirs(f"data/{GPU}/results/flash_attn")
        df.to_pickle(fp)


if __name__ == "__main__":
    main()
