"""
Block Pointer (Experimental)
============================
This tutorial will guide you through writing a matrix multiplication algorithm that utilizes block pointer semantics.
These semantics are more friendly for Triton to optimize and can result in better performance on specific hardware.
Note that this feature is still experimental and may change in the future.

"""
import random
import numpy as np

import torch

import triton
import triton.language as tl

from fgk.jit import search

from absl import app
from absl import flags

# YAPF: disable
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
flags.DEFINE_integer("wl", 4096, "")


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True

    M = FLAGS.wl
    a = torch.randn((M, 2*M), device='cuda', dtype=torch.float16)
    b = torch.randn((2*M, M), device='cuda', dtype=torch.float16)
    @search(
        total_flops=M*M*(2*2*M-1),
        seed=FLAGS.seed,
        save_suffix=str(M),
        save_dir='matmul',
    )
    def matmul_kernel_with_block_pointers(
            # Pointers to matrices
            a_ptr,
            b_ptr,
            c_ptr,
            # Matrix dimensions
            M,
            N,
            K,
            # The stride variables represent how much to increase the ptr by when moving by 1
            # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
            # by to get the element one row down (A has M rows).
            stride_am,
            stride_ak,  #
            stride_bk,
            stride_bn,  #
            stride_cm,
            stride_cn,
            # Meta-parameters
            BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_N: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr,
            GROUP_SIZE_M: tl.constexpr):
        """Kernel for computing the matmul C = A x B.
        A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse.
        # See the matrix multiplication tutorial for details.
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ----------------------------------------------------------
        # Create block pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction and accumulate.
        # See above `Make a Block Pointer` section for details.
        a_block_ptr = tl.make_block_ptr(base=a_ptr,
                                        shape=(M, K),
                                        strides=(stride_am, stride_ak),
                                        offsets=(pid_m * BLOCK_SIZE_M, 0),
                                        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                        order=(1, 0))
        b_block_ptr = tl.make_block_ptr(base=b_ptr,
                                        shape=(K, N),
                                        strides=(stride_bk, stride_bn),
                                        offsets=(0, pid_n * BLOCK_SIZE_N),
                                        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                        order=(1, 0))

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block.
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_SIZE_K):
            # Load with boundary checks, no need to calculate the mask manually.
            # For better performance, you may remove some axis from the boundary
            # check, if you can guarantee that the access is always in-bound in
            # that axis.
            # See above `Load/Store a Block Pointer` section for details.
            a = tl.load(a_block_ptr, boundary_check=(0, 1))
            b = tl.load(b_block_ptr, boundary_check=(0, 1))
            # We accumulate along the K dimension.
            accumulator += tl.dot(a, b)
            # Advance the block pointer to the next K block.
            # See above `Advance a Block Pointer` section for details.
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
        c = accumulator.to(tl.float16)

        # ----------------------------------------------------------------
        # Write back the block of the output matrix C with boundary checks.
        # See above `Load/Store a Block Pointer` section for details.
        c_block_ptr = tl.make_block_ptr(base=c_ptr,
                                        shape=(M, N),
                                        strides=(stride_cm, stride_cn),
                                        offsets=(pid_m * BLOCK_SIZE_M,
                                                pid_n * BLOCK_SIZE_N),
                                        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
                                        order=(1, 0))
        tl.store(c_block_ptr, c, boundary_check=(0, 1))


    # We can now create a convenience wrapper function that only takes two input tensors,
    # and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.
    def matmul(a, b):
        # Check constraints.
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        assert a.is_contiguous(), "Matrix A must be contiguous"
        assert b.is_contiguous(), "Matrix B must be contiguous"
        M, K = a.shape
        K, N = b.shape
        # Allocates output.
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        # 1D launch kernel where each block gets its own program.

        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),

        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(
            N, META['BLOCK_SIZE_N']), )
        matmul_kernel_with_block_pointers[grid](
            a, b, c,  #
            M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=256,
            BLOCK_SIZE_K=64,
            GROUP_SIZE_M=8,
            # num_stages=4,  # ampere
            num_stages=2,   # turing
            num_warps=4,
            )
        return c


    # %%
    # Unit Test
    # ---------
    #
    # Still we can test our matrix multiplication with block pointers against a native torch implementation (i.e., cuBLAS).

    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")


if __name__ == "__main__":
    app.run(main)
