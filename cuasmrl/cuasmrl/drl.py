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
from cuasmrl.ppo import env_loop, inference
from cuasmrl.utils.logger import get_logger
from cuasmrl.utils.record import save_data, read_data
from cuasmrl.verify import test_via_cubin, gen_test_samples

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

    if config.train == 1:
        is_inference = False
    else:
        is_inference = True

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

    # TODO gen static test samples
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

        #
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
        is_inference,
    )()

    # ===== run =====
    if is_inference:
        inference(env, config)
    else:
        env_loop(env, config)
