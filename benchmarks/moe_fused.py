
import os
import random
import numpy as np
import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
import triton
import triton.language as tl

from cuasmrl.jit import jit
from cuasmrl.autotuner import autotune as fgk_autotune
from cuasmrl.utils.gpu_utils import get_gpu_name, get_gpu_cc

# install vllm
from vllm._C import ops
import vllm._moe_C as moe_kernels

# YAPF: disable

@dataclass
class Config:
    # Kernel
    default_out_path: str = "data"
    seed: int = 1337
    n_tests: int = 2
    load: Optional[str] = None
    bench: int = 0

    # Workload
    m: int = 512
    n: int = 7168
    k: int = 4096
    e: int = 8
    topk: int = 2

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

    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=7168)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--e", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)

    parser.add_argument("-t", "--train", type=int, dest="train", default=1)
    parser.add_argument("-l", "--log", type=int, dest="log", default=1)
    parser.add_argument("--verbose", type=int, default=0)

    parser.add_argument("--env_id", type=str, default='cuasmenv-v0')
    parser.add_argument("--num_iterations", type=int, default=int(1e3))
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
def tt_moe(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N, K, EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn, stride_weight, stride_token_id,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    # -----------------------------------------------------------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token * stride_weight,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def moe_align_block_size(topk_ids: torch.Tensor, block_size: int,
                         num_experts: int):
    sorted_ids = torch.empty(
        (topk_ids.numel() + num_experts * (block_size - 1), ),
        dtype=torch.int32,
        device=topk_ids.device)
    expert_ids = torch.empty((topk_ids.numel() + num_experts, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)

    ops.moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
                             expert_ids, num_tokens_post_pad)
    return sorted_ids, expert_ids, num_tokens_post_pad


def invoke_fused_moe_kernel(kernel,
                            drl_config,
                             A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                            topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool, top_k: int, config: dict):

    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META[
        'BLOCK_SIZE_M']) * triton.cdiv(B.shape[1], META['BLOCK_SIZE_N']), )

    # print(f"Base {config}\n")
    if drl_config.tt == 0:
        kernel[grid](
            A, B, C,
            topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
            B.shape[1], B.shape[2],
            sorted_token_ids.shape[0],
            topk_ids.numel(),
            A.stride(0), A.stride(1), B.stride(0), B.stride(2), B.stride(1), C.stride(1), C.stride(2),
            topk_weights.stride(1),
            sorted_token_ids.stride(0),
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16,

            # gh512: need a hint
            load_dir = drl_config.load_dir, 
            # **config,
        )
    elif drl_config.tt == 1:
        kernel[grid](
            A, B, C,
            topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
            B.shape[1], B.shape[2],
            sorted_token_ids.shape[0],
            topk_ids.numel(),
            A.stride(0), A.stride(1), B.stride(0), B.stride(2), B.stride(1), C.stride(1), C.stride(2),
            topk_weights.stride(1),
            sorted_token_ids.stride(0),
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16,
            **config,
        )


def fused_moe(
        kernel,
        drl_config,
        hidden_states: torch.Tensor,
              gate,
              w1: torch.Tensor,
              w2: torch.Tensor,
              topk,
              renormalize=True,
              inplace=False,
            ):

    # Check constraints.
    assert hidden_states.shape[1] == w1.shape[2], "Incompatible dimensions"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    M, _ = hidden_states.shape
    E, N, _ = w1.shape

    config = {
        'BLOCK_SIZE_M': 64,
        'BLOCK_SIZE_N': 64,
        'BLOCK_SIZE_K': 32,
        'GROUP_SIZE_M': 8
    }

    # if topk_ids.numel() <= w1.shape[0]:
    if M * topk <= w1.shape[0]:
        config = {
            'BLOCK_SIZE_M': 16,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 1
        }

    # NOTE: statically allocate route parameter
    vllm_topk_weights = torch.empty(M,
                                    topk,
                                    dtype=torch.float32,
                                    device=hidden_states.device)
    vllm_topk_ids = torch.empty(M,
                                topk,
                                dtype=torch.int32,
                                device=hidden_states.device)
    vllm_token_expert_indicies = torch.empty(M,
                                             topk,
                                             dtype=torch.int32,
                                             device=hidden_states.device)

    gating_output = hidden_states @ gate
    moe_kernels.topk_softmax(
        vllm_topk_weights,
        vllm_topk_ids,
        vllm_token_expert_indicies,
        gating_output.float(),  # TODO(woosuk): Optimize this.
    )
    del vllm_token_expert_indicies  # Not used. Will be used in the future.
    if renormalize:
        vllm_topk_weights = vllm_topk_weights / vllm_topk_weights.sum(
            dim=-1, keepdim=True)

    # fused moe op
    intermediate_cache1 = torch.empty((M, vllm_topk_ids.shape[1], N),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache2 = torch.empty((M * vllm_topk_ids.shape[1], N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache3 = torch.empty((M, vllm_topk_ids.shape[1], w2.shape[1]),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        vllm_topk_ids, config['BLOCK_SIZE_M'], E)

    invoke_fused_moe_kernel(kernel, drl_config, hidden_states, w1, intermediate_cache1,
                            vllm_topk_weights, vllm_topk_ids, sorted_token_ids,
                            expert_ids, num_tokens_post_padded, False,
                            vllm_topk_ids.shape[1], config)

    # ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

    # invoke_fused_moe_kernel(kernel, drl_config, intermediate_cache2, w2, intermediate_cache3,
    #                         vllm_topk_weights, vllm_topk_ids, sorted_token_ids,
    #                         expert_ids, num_tokens_post_padded, True, 1,
    #                         config)

    # if inplace:
    #     return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
    #                      dim=1,
    #                      out=hidden_states)
    # return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
    #                  dim=1)
    return intermediate_cache1


if __name__ == '__main__':

    config = parse_args()

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True

    m, k, e, n = config.m, config.k, config.e, config.n
    topk = config.topk
    dtype = torch.float16
    a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    gate = torch.randn((k, e), device='cuda', dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10

    config.total_flops = 2 * m * n * k 
    config.save_dir = f'{GPU}/moe/{m}_{k}_{e}_{n}_{topk}'

    if config.load is None:
        load_dir = None
    elif config.load == "auto":
        load_dir = f'data/{GPU}/moe/{m}_{k}_{e}_{n}_{topk}'
    else:
        load_dir = config.load
    config.load_dir = load_dir

    triton_config = {
        'BLOCK_SIZE_M': 64,
        'BLOCK_SIZE_N': 64,
        'BLOCK_SIZE_K': 32,
        'GROUP_SIZE_M': 8
    }
    if m * topk <= w1.shape[0]:
        triton_config = {
            'BLOCK_SIZE_M': 16,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 1
    }

    @fgk_autotune(
    configs=[
        triton.Config(triton_config),
    ],
    key=['top_k'],
    ret_ptr=2,
    drl_config=config,
    )
    @jit
    def cuasmrl_moe_kernel(
        a_ptr, b_ptr, c_ptr,
        topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
        num_tokens_post_padded_ptr,
        # Matrix dimensions
        N, K, EM,
        num_valid_tokens,
        stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn, stride_weight, stride_token_id,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
    ):
        # -----------------------------------------------------------
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ----------------------------------------------------------
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return

        offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
        token_mask = offs_token < num_valid_tokens

        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                        offs_k[None, :] * stride_ak)

        off_experts = tl.load(expert_ids_ptr + pid_m)
        b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                    offs_bn[None, :] * stride_bn)

        # -----------------------------------------------------------
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            a = tl.load(a_ptrs,
                        mask=token_mask[:, None] &
                        (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                        other=0.0)
            b = tl.load(b_ptrs,
                        mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                        other=0.0)
            # We accumulate along the K dimension.
            accumulator += tl.dot(a, b)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(topk_weights_ptr + offs_token * stride_weight,
                                mask=token_mask,
                                other=0)
            accumulator = accumulator * moe_weight[:, None]

        accumulator = accumulator.to(compute_type)
        # -----------------------------------------------------------
        # Write back the block of the output
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
            None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    # invoke
    config.tt = 0
    out = fused_moe(cuasmrl_moe_kernel, config, a, gate, w1, w2, topk, True, False)

    # if invoke triton
    config.tt = 1
    ref = fused_moe(tt_moe, config, a, gate, w1, w2, topk, True, False)

    assert torch.allclose(out, ref, atol=1e-2, rtol=0)
    print('TEST PASSED')