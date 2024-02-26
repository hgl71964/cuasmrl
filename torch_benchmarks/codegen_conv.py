from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
import numpy as np
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
import torch._inductor.kernel.conv

meta0 = {
    'KERNEL_H': 3,
    'KERNEL_W': 3,
    'STRIDE_H': 1,
    'STRIDE_W': 1,
    'PADDING_H': 1,
    'PADDING_W': 1,
    'GROUPS': 1,
    'UNROLL': False,
    'ALLOW_TF32': True,
    'BLOCK_M': 1024,
    'BLOCK_N': 16,
    'BLOCK_K': 16
}

import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

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
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(primals_2, (6, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_3, (6, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)  # no-op to ensure context
        buf1 = empty_strided((32, 6, 224, 224), (301056, 50176, 224, 1),
                             device='cuda',
                             dtype=torch.float32)
        # Source Nodes: [conv2d], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        # conv_kernel.run(primals_1, primals_2, primals_3, buf1, grid=torch._inductor.kernel.conv.conv_grid(32, 6, 224, 224, meta0), stream=stream0)

        grid = torch._inductor.kernel.conv.conv_grid(32, 6, 224, 224, meta0)
        kernel[grid](primals_1,
                     primals_2,
                     primals_3,
                     buf1,
                     KERNEL_H=3,
                     KERNEL_W=3,
                     STRIDE_H=1,
                     STRIDE_W=1,
                     PADDING_H=1,
                     PADDING_W=1,
                     GROUPS=1,
                     UNROLL=False,
                     ALLOW_TF32=True,
                     BLOCK_M=1024,
                     BLOCK_N=16,
                     BLOCK_K=16,
                     stream=stream0,
                     num_stages=1,
                     num_warps=8)
        del primals_3
        return (
            buf1,
            primals_1,
            primals_2,
        )


def run(kernel, times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 224, 224), (150528, 50176, 224, 1),
                             device='cuda:0',
                             dtype=torch.float32)
    primals_2 = rand_strided((6, 3, 3, 3), (27, 9, 3, 1),
                             device='cuda:0',
                             dtype=torch.float32)
    primals_3 = rand_strided((6, ), (1, ),
                             device='cuda:0',
                             dtype=torch.float32)
    out, _, _ = call([primals_1, primals_2, primals_3], kernel)
    ref = torch.nn.Conv2d(3, 6, kernel_size=3, padding=1)(primals_1)
    assert torch.allclose(out, ref, atol=1e-2, rtol=0)
    # return print_performance(lambda: call([primals_1, primals_2, primals_3]), times=times, repeat=repeat)


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True

    # YAPF: disable
    # TODO fail to assemble at A100
    # @template(num_stages=1, num_warps=8, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]})
    # @triton.jit
    @jit(
        total_flops=1e9,  # just to make it working
        seed=FLAGS.seed,
        save_dir='conv3x3',
    )
    def conv_kernel(
        arg_X, arg_W, in_ptr2, out_ptr1,  # 
        KERNEL_H: tl.constexpr,
        KERNEL_W: tl.constexpr,
        STRIDE_H: tl.constexpr,
        STRIDE_W: tl.constexpr,
        PADDING_H: tl.constexpr,
        PADDING_W: tl.constexpr,
        GROUPS: tl.constexpr,
        UNROLL: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):

        X = arg_X
        W = arg_W

        # Tensor dimensions
        BATCH = 32
        IN_C = 3
        IN_H = 224
        IN_W = 224
        OUT_C = 6
        OUT_H = 224
        OUT_W = 224

        # Strides:
        stride_xn = 150528
        stride_xc = 50176
        stride_xh = 224
        stride_xw = 1
        stride_wc_out = 27
        stride_wc_in = 9
        stride_wh = 3
        stride_ww = 1

        nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        idx_y_w = nhw % OUT_W
        nh = nhw // OUT_W
        idx_y_h = nh % OUT_H
        idx_n = nh // OUT_H
        idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

        group = 0
        GROUP_IN_C = IN_C
        GROUP_OUT_C = OUT_C

        x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:,
                                                                          None]
        w_base = (W + (group * stride_wc_out * GROUP_OUT_C +
                       idx_y_c * stride_wc_out)[None, :])

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Could be simplified, but slightly slower:
        # for i in range(KERNEL_H):
        #     for j in range(KERNEL_W):
        #         for k in range(0, GROUP_IN_C, BLOCK_K):
        BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
        for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
            k = (ijk % BLOCK_K_COUNT) * BLOCK_K
            ij = ijk // BLOCK_K_COUNT
            i = ij // KERNEL_W
            j = ij % KERNEL_W

            idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
            idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
            idx_x_c = tl.arange(0, BLOCK_K) + k

            x_ptrs = x_base + ((idx_x_h * stride_xh)[:, None] +
                               (idx_x_w * stride_xw)[:, None] +
                               (idx_x_c * stride_xc)[None, :])
            mask_x = ((idx_n < BATCH)[:, None]
                      & (idx_x_h >= 0)[:, None]
                      & (idx_x_h < IN_H)[:, None]
                      & (idx_x_w >= 0)[:, None]
                      & (idx_x_w < IN_W)[:, None]
                      & (idx_x_c < GROUP_IN_C)[None, :])
            matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

            w_ptrs = w_base + ((idx_x_c * stride_wc_in)[:, None] +
                               (i * stride_wh) + (j * stride_ww))
            mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :]
                                                        < GROUP_OUT_C)
            matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
            acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)

        mask = ((idx_n < BATCH)[:, None]
                & (idx_y_h < OUT_H)[:, None]
                & (idx_y_w < OUT_W)[:, None]
                & (idx_y_c < GROUP_OUT_C)[None, :])
        idx_n = idx_n[:, None]
        idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
        idx_h = idx_y_h[:, None]
        idx_w = idx_y_w[:, None]

        # inductor generates a suffix
        xindex = idx_w + (224 * idx_h) + (50176 * idx_c) + (301056 * idx_n)
        x5 = xindex % 50176
        tmp0 = tl.load(in_ptr2 + (tl.broadcast_to(idx_c, mask.shape)),
                       mask,
                       eviction_policy='evict_last')
        tmp1 = acc + tmp0
        tl.store(out_ptr1 + (x5 + (50176 * idx_c) + (301056 * idx_n)), tmp1,
                 mask)
    # YAPF: enable

    run(conv_kernel)


if __name__ == "__main__":
    # from torch._inductor.wrapper_benchmark import compiled_module_main
    # compiled_module_main('None', benchmark_compiled_module)
    app.run(main)
