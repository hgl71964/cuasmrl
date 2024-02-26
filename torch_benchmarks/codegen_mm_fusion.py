from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor

import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
import torch._inductor.kernel.mm_common

meta0 = {
    'GROUP_M': 8,
    'EVEN_K': True,
    'ALLOW_TF32': True,
    'ACC_TYPE': 'tl.float32',
    'B_PROLOGUE_CAST_TYPE': None,
    'BLOCK_M': 64,
    'BLOCK_N': 128,
    'BLOCK_K': 32
}

from fgk.jit import jit

from absl import app
from absl import flags

FLAGS = flags.FLAGS
# kernel
flags.DEFINE_string("default_out_path", "data", "output dir")
flags.DEFINE_integer("seed", 1337, "")
flags.DEFINE_integer("test_sample", 10, "")
flags.DEFINE_integer("n_choices", 1, "+-n choices")
# sa
flags.DEFINE_integer("max_iterations", 1000, "")
flags.DEFINE_float("temperature", 0.4, "")
flags.DEFINE_float("cooling_rate", 0.003, "")
flags.DEFINE_float("noise_factor", 0.1, "")
flags.DEFINE_string("policy", "single", "mutation policy; single or all")
# genetic
flags.DEFINE_integer("population_size", 100, "")
flags.DEFINE_integer("generations", 50, "")
flags.DEFINE_float("mutation_rate", 0.1, "")
flags.DEFINE_integer("tournament_size", 5, "")
#workload
flags.DEFINE_integer("wl", 4096, "workload of chosen")


def call(args, kernel):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 9216), (9216, 1))
    assert_size_stride(arg1_1, (9216, 4096), (4096, 1))
    assert_size_stride(arg2_1, (128, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)  # no-op to ensure context
        buf1 = empty_strided((128, 4096), (4096, 1),
                             device='cuda',
                             dtype=torch.float16)
        # Source Nodes: [iadd, relu], Original ATen: [aten.add, aten.relu]
        stream0 = get_cuda_stream(0)
        grid = torch._inductor.kernel.mm_common.mm_grid(128, 4096, meta0)
        kernel[grid](
            arg2_1,
            arg0_1,
            arg1_1,
            buf1,
            stream=stream0,
            num_stages=4,
            num_warps=8,
        )
        del arg0_1
        del arg1_1
        del arg2_1
        return (buf1, )


def benchmark_compiled_module(kernel, times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((128, 9216), (9216, 1),
                          device='cuda:0',
                          dtype=torch.float16)
    arg1_1 = rand_strided((9216, 4096), (4096, 1),
                          device='cuda:0',
                          dtype=torch.float16)
    arg2_1 = rand_strided((128, 4096), (4096, 1),
                          device='cuda:0',
                          dtype=torch.float16)
    out = call([arg0_1, arg1_1, arg2_1], kernel)
    # return print_performance(lambda: call([arg0_1, arg1_1, arg2_1]), times=times, repeat=repeat)


def main(_):
    # @template(num_stages=4, num_warps=8, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]})
    # @triton.jit
    @jit(
        total_flops=1e9,  # just to make it working
        seed=FLAGS.seed,
        # save_suffix=str(N),
        save_dir='mm_fusion',
    )
    def triton_(in_ptr0, arg_A, arg_B, out_ptr1):
        GROUP_M: tl.constexpr = 8
        EVEN_K: tl.constexpr = True
        ALLOW_TF32: tl.constexpr = True
        ACC_TYPE: tl.constexpr = tl.float32
        B_PROLOGUE_CAST_TYPE: tl.constexpr = None
        BLOCK_M: tl.constexpr = 64
        BLOCK_N: tl.constexpr = 128
        BLOCK_K: tl.constexpr = 32

        A = arg_A
        B = arg_B

        M = 128
        N = 4096
        K = 9216
        if M * N == 0:
            # early exit due to zero-size input(s)
            return
        stride_am = 9216
        stride_ak = 1
        stride_bk = 4096
        stride_bn = 1

        # based on triton.ops.matmul
        pid = tl.program_id(0)
        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        grid_n = (N + BLOCK_N - 1) // BLOCK_N

        # re-order program ID for better L2 performance
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // (group_size)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
        B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
        for k in range(K, 0, -BLOCK_K):
            if EVEN_K:
                a = tl.load(A)
                b = tl.load(B)
            else:
                a = tl.load(A, mask=rk[None, :] < k, other=0.)
                b = tl.load(B, mask=rk[:, None] < k, other=0.)
            if B_PROLOGUE_CAST_TYPE is not None:
                b = b.to(B_PROLOGUE_CAST_TYPE)
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A += BLOCK_K * stride_ak
            B += BLOCK_K * stride_bk

        # rematerialize rm and rn to save registers
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)

        # inductor generates a suffix
        xindex = idx_n + (4096 * idx_m)
        tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n +
                                                  (4096 * idx_m), mask.shape)),
                       mask,
                       eviction_policy='evict_last').to(tl.float32)
        tmp1 = acc + tmp0
        tmp2 = triton_helpers.maximum(0, tmp1)
        tl.store(out_ptr1 + (tl.broadcast_to(xindex, mask.shape)), tmp2, mask)

    benchmark_compiled_module(triton_)


if __name__ == "__main__":
    # from torch._inductor.wrapper_benchmark import compiled_module_main
    # compiled_module_main('None', benchmark_compiled_module)
    app.run(main)
