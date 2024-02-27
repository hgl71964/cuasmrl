import os
import time
from copy import deepcopy

import numpy as np
import torch

# mutation
from cuasmrl.utils.logger import get_logger

from cuasmrl.compiler import CompiledKernel as fgk_CompiledKernel

logger = get_logger(__name__)


def gen_test_samples(
    bin,
    non_constexpr_arg_values,
    grid_0,
    grid_1,
    grid_2,
    stream,
    launch_enter_hook,
    launch_exit_hook,
    n_test_samples,
    ret_ptr,
) -> list[dict]:
    test_samples = []
    for t in range(n_test_samples):
        # generate test sample
        test_list = []
        for i, inp in enumerate(non_constexpr_arg_values):
            arg = None

            if isinstance(inp, torch.Tensor):
                if i == ret_ptr:
                    arg = torch.empty_like(inp)
                else:
                    # arg = torch.randn_like(inp).uniform_(0, 1)
                    arg = torch.randn_like(inp)
            else:
                arg = deepcopy(inp)

            test_list.append(arg)

        # call
        bin.c_wrapper(
            grid_0,
            grid_1,
            grid_2,
            bin.num_warps,
            bin.num_ctas,
            bin.clusterDims[0],
            bin.clusterDims[1],
            bin.clusterDims[2],
            bin.shared,
            stream,
            bin.cu_function,
            launch_enter_hook,
            launch_exit_hook,
            bin,
            *bin.assemble_tensormap_to_arg(test_list),
        )
        test_samples.append(test_list)

    return test_samples


def e2e_test(
    bin,
    grid_0,
    grid_1,
    grid_2,
    stream,
    launch_enter_hook,
    launch_exit_hook,
    ret_ptr,
    test_samples,
) -> list[bool]:
    oks = []
    for t, test_sample in enumerate(test_samples):

        test_list = []
        ref = None
        for idx, inp in enumerate(test_sample):
            if isinstance(inp, torch.Tensor) and idx == ret_ptr:
                ref = inp
                arg = torch.empty_like(ref)
                out_buffer = arg
            else:
                arg = inp

            test_list.append(arg)

        bin.c_wrapper(
            grid_0,
            grid_1,
            grid_2,
            bin.num_warps,
            bin.num_ctas,
            bin.clusterDims[0],
            bin.clusterDims[1],
            bin.clusterDims[2],
            bin.shared,
            stream,
            bin.cu_function,
            launch_enter_hook,
            launch_exit_hook,
            bin,
            *bin.assemble_tensormap_to_arg(test_list),
        )

        # TODO: atm we only consider one output from kernel
        if torch.allclose(ref, out_buffer, atol=1e-2, rtol=0):
            oks.append(True)
        else:
            oks.append(False)

    return oks


def test_via_cubin(
    # compile args
    so_path,
    metadata,
    asm,

    # args
    args,
    sig_key,
    non_constexpr_arg_values,
    ret_ptr,
    test_inputs,
    test_outputs,

    # kernel args
    grid_0,
    grid_1,
    grid_2,
    stream,  # 
    enter_hook,
    exit_hook,
    cubin,
    n_test_samples,
    test_batch_size=1,
):
    # return True
    bin = fgk_CompiledKernel(so_path, metadata, asm)

    # use hint to generate test cases
    if ret_ptr is not None:
        okss = []
        for _ in range(0, n_test_samples, test_batch_size):
            test_samples = gen_test_samples(
                bin,
                non_constexpr_arg_values,
                grid_0,
                grid_1,
                grid_2,
                stream,
                enter_hook,
                exit_hook,
                # n_test_samples,
                test_batch_size,
                ret_ptr,
            )
            opt_asm = {
                'cubin': cubin,
            }
            opt_bin = fgk_CompiledKernel(so_path, metadata, opt_asm)

            oks = e2e_test(
                opt_bin,
                grid_0,
                grid_1,
                grid_2,
                stream,
                enter_hook,
                exit_hook,
                ret_ptr,
                test_samples,
            )
            okss.extend(oks)
            torch.cuda.empty_cache()  # free test memory for one test

            if not np.all(oks):
                # early stop
                break

        passes = sum(okss)
        total = len(okss)
        if np.all(okss):
            logger.info(f"✅ kernel verified for {total} test samples")
        else:
            logger.error(f"❌ kernel fail; only {passes}/{total} passes")

        all_ok = np.all(okss)
        return all_ok

    else:
        raise NotImplementedError('impl custom test verifier')
