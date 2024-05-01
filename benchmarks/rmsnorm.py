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

    parser.add_argument("--Z", type=int, dest="Z", default=1)
    parser.add_argument("--H", type=int, dest="H", default=64)
    parser.add_argument("--wl", type=int, default=128)
    parser.add_argument("--dh", type=int, dest="D_HEAD", default=128)

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
def rmsnorm_triton(x_ptr, rms_w_ptr, output_ptr,
                   stride_x_batch, stride_x_m, stride_x_k,
                   stride_rms_w,
                   stride_out_batch, stride_out_m, stride_out_k,
                   N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_N = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0)
        var += tl.math.pow(x.to(tl.float32), 2)

    var = tl.sum(var, axis=0) / N_SIZE
    rstd = tl.math.rsqrt(var + eps)

    # multiply by weight and add bias
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        rms_w = tl.load(rms_w_ptr + offs_n * stride_rms_w, mask=x_ptr_mask)

        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        out = x_hat * rms_w
        out_off = pid_batch * stride_out_batch + pid_m * stride_out_m + offs_n * stride_out_k
        tl.store(output_ptr + out_off, out, mask=x_ptr_mask)


def call_tt(x, rms_w, eps=1e-6):
    batch, M, K = x.shape
    assert rms_w.shape[-1] == K
    out = torch.empty_like(x)
    rmsnorm_triton[(batch, M,)](x, rms_w, out,
                                *x.stride(),
                                *rms_w.stride(),
                                *out.stride(),
                                N_SIZE=K, eps=eps, BLOCK_N_SIZE=1024,
                                )
    return out

def call(kernel, load_dir, x, rms_w, eps=1e-6):
    batch, M, K = x.shape
    assert rms_w.shape[-1] == K
    out = torch.empty_like(x)
    kernel[(batch, M,)](x, rms_w, out,
                                *x.stride(),
                                *rms_w.stride(),
                                *out.stride(),
                                # N_SIZE=K, eps=eps, BLOCK_N_SIZE=1024,
                                load_dir=load_dir,
                                )
    return out


if __name__ == '__main__':
    drl_config = parse_args()

    random.seed(drl_config.seed)
    np.random.seed(drl_config.seed)
    torch.manual_seed(drl_config.seed)

    batch, heads, seq_len, dim = drl_config.Z, drl_config.H, drl_config.wl, drl_config.D_HEAD
    K=heads*dim

    embeddings_load = torch.randn([batch, seq_len, heads * dim], dtype=torch.float16, device="cuda")
    rms_weights = torch.randn([heads * dim], dtype=torch.float16, device="cuda") * 0.2
    q_weights_load = torch.randn([heads * dim, heads * dim], dtype=torch.float16, device="cuda") * 0.2


    drl_config.total_flops = batch * seq_len * heads * dim
    drl_config.save_dir = f'{GPU}/rmsnorm/{batch}_{heads}_{seq_len}_{dim}'

    if drl_config.load is None:
        load_dir = None
    elif drl_config.load == "auto":
        load_dir = f'data/{GPU}/rmsnorm/{batch}_{heads}_{seq_len}_{dim}'
    else:
        load_dir = drl_config.load
    

    @fgk_autotune(
        configs=[
		triton.Config({'N_SIZE': K, 'eps': 1e-6, 'BLOCK_N_SIZE':32}, num_stages=4, num_warps=4),
            
    ],
        key=['N_SIZE'],
        ret_ptr=2,
        drl_config=drl_config,
    )
    @jit
    def _cuasmrl(x_ptr, rms_w_ptr, output_ptr,
                    stride_x_batch, stride_x_m, stride_x_k,
                    stride_rms_w,
                    stride_out_batch, stride_out_m, stride_out_k,
                    N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
        pid_batch = tl.program_id(0)
        pid_m = tl.program_id(1)

        offs_m = pid_batch * stride_x_batch + pid_m * stride_x_m
        block_N = tl.arange(0, BLOCK_N_SIZE)
        var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
        for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
            offs_n = block_n_start_idx + block_N
            x_ptr_mask = offs_n < N_SIZE
            x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0)
            var += tl.math.pow(x.to(tl.float32), 2)

        var = tl.sum(var, axis=0) / N_SIZE
        rstd = tl.math.rsqrt(var + eps)

        # multiply by weight and add bias
        for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
            offs_n = block_n_start_idx + block_N
            x_ptr_mask = offs_n < N_SIZE
            rms_w = tl.load(rms_w_ptr + offs_n * stride_rms_w, mask=x_ptr_mask)

            x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0).to(tl.float32)
            x_hat = x * rstd
            out = x_hat * rms_w
            out_off = pid_batch * stride_out_batch + pid_m * stride_out_m + offs_n * stride_out_k
            tl.store(output_ptr + out_off, out, mask=x_ptr_mask)
    
    call(_cuasmrl, load_dir, embeddings_load, rms_weights)

    if drl_config.tt:
        out_rms_triton = call_tt(x=embeddings_load, rms_w=rms_weights)

    if bool(drl_config.bench):
        print('BENCH...')
        torch.cuda.synchronize()

        ms = do_bench(lambda: call(_cuasmrl, load_dir, embeddings_load, rms_weights), warmup=100, rep=100)
        ms_tt = do_bench(lambda: call_tt(x=embeddings_load, rms_w=rms_weights), warmup=100, rep=100)

        data = {
            'cuasmrl': ms,
            'tt': ms_tt,
        }
        print(data)

        fp = f"data/{GPU}/rmsnorm/{batch}_{heads}_{seq_len}_{dim}/bench_{drl_config.seed}.pkl"
        with open(fp, 'wb') as f:
            pickle.dump(data, f)