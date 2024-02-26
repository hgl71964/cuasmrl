import os
import math
import random
import tempfile
import time
from functools import partial
from copy import deepcopy

import numpy as np
import torch

# asm (add CuAsm to PYTHONPATH!)
from CuAsm.CubinFile import CubinFile

# mutation
from cuasmrl.backend import make_env, MutationEngine
from cuasmrl.ppo import env_loop
from cuasmrl.utils.logger import get_logger
from cuasmrl.utils.record import save_data, read_data
from cuasmrl.verify import test_via_cubin

logger = get_logger(__name__)


def run_drl(
    # kernel
    bin,
    so_path,
    metadata,
    asm,
    ret_ptr,
    args,
    sig_key,
    non_constexpr_arg_values,
    grid_0,
    grid_1,
    grid_2,
    stream,
    launch_enter_hook,
    launch_exit_hook,

    # drl config
    config,
):
    # get initial cubin and asm (the initial have to file IO)
    with tempfile.NamedTemporaryFile(mode='wb', delete=True) as temp_file:
        cubin = bin.asm['cubin']
        temp_file.write(cubin)
        # Ensure data is written to the file before reading it
        temp_file.flush()
        temp_file.seek(0)

        time.sleep(1)
        cf = CubinFile(temp_file.name)

    # ===== backend env =====
    eng = MutationEngine(
        bin,
        cf,
        grid_0,
        grid_1,
        grid_2,
        stream,
        launch_enter_hook,
        launch_exit_hook,
        non_constexpr_arg_values,
        config.total_flops,
    )

    test_fn = partial(
        test_via_cubin,
        so_path,
        metadata,
        asm,
        args,
        sig_key,
        non_constexpr_arg_values,
        ret_ptr,
        None,
        None,
        grid_0,
        grid_1,
        grid_2,
        stream,
        launch_enter_hook,
        launch_exit_hook,
    )
    eng.test_fn = test_fn
    env = make_env(
        config.env_id,
        eng,
        config,
    )

    # ===== run =====
    _t1 = time.perf_counter()
    best_solution, best_perf = env_loop(env, config)

    _t2 = time.perf_counter()
    hours = (_t2 - _t1) / 3600

    final_perf = best_perf
    _ = eng.assemble(
        best_solution
    )  # if illegal memory access, this gives error, but cubin is valid

    logger.info(
        f'Performance: {final_perf:.2f}; init perf: {init_perf:.2f}; Search time: {hours:.2f}h'
    )
    logger.info(
        f'improvement: {(final_perf - init_perf) / init_perf * 100:.2f}%')

    # ===== save =====
    path = save_data(
        bin,
        final_perf,
        init_perf,
        hours,
        args,
        sig_key,
        non_constexpr_arg_values,
        config.seed,
        config.save_suffix,
        config.save_dir,
        algo='ppo',
    )
    return path
