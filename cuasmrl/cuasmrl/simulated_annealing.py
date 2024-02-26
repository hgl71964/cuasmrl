import os
import math
import random
import tempfile
import time
from functools import partial
from copy import deepcopy

import multiprocessing

import numpy as np
import torch

# asm (add CuAsm to PYTHONPATH!)
from CuAsm.CubinFile import CubinFile

# mutation
from cuasmrl.mutator import MutationEngine
from cuasmrl.sample import Sample
from cuasmrl.utils.logger import get_logger
from cuasmrl.utils.record import save_data, read_data

from cuasmrl.verify import test_via_cubin

logger = get_logger(__name__)


class SimulatedSample(Sample):

    def apply(self, index, action):
        lineno = self.candidates[index]
        if action == -1:
            self.kernel_section[lineno - 1], self.kernel_section[
                lineno] = self.kernel_section[lineno], self.kernel_section[
                    lineno - 1]
            self.candidates[index] -= 1
        elif action == 1:
            self.kernel_section[lineno], self.kernel_section[
                lineno +
                1] = self.kernel_section[lineno +
                                         1], self.kernel_section[lineno]
            self.candidates[index] += 1
        elif action == 0:
            pass
        else:
            assert False, f'invalid action: {action}'


def generate_neighbor(sample: SimulatedSample, n_choices, policy):
    mutable = sample.get_mutable()
    if policy == 'single':
        index = random.randint(0, len(mutable) - 1)
        action = random.choice([-1, 1])
        indexes = [index]
        actions = [action]
    elif policy == 'all':
        n = len(mutable)
        indexes = [i for i in range(n)]
        actions = [
            random.choice(range(-n_choices, n_choices)) for _ in range(n)
        ]
    else:
        raise RuntimeError(f'invalid policy: {policy}')

    neighbor = SimulatedSample(sample.kernel_section, sample.engine)
    neighbor.candidates = deepcopy(mutable)
    neighbor.dims = sample.dims
    neighbor.apply_all(indexes, actions)
    return neighbor


def acceptance_probability(old_fitness, new_fitness, temperature,
                           noise_factor):
    noise = random.uniform(-noise_factor, noise_factor)
    adjusted_difference = new_fitness - old_fitness + noise

    if adjusted_difference > 0:
        return 1.0

    return math.exp(adjusted_difference / temperature)


def simulated_annealing(
    initial_solution: SimulatedSample,
    init_fitness,
    n_choices,
    max_iterations,
    temperature,
    cooling_rate,
    policy,
    noise_factor,
    eng: MutationEngine,
    test_fn,
) -> SimulatedSample:
    current_solution = initial_solution
    current_fitness = init_fitness
    best_solution = current_solution
    best_fitness = current_fitness
    cnt = 1

    while temperature > 0.05 and cnt < max_iterations:
        new_solution = generate_neighbor(current_solution, n_choices, policy)
        new_fitness, cubin = eng.get_perf(new_solution)
        test_ok = True
        if new_fitness > 0:
            if not test_fn(cubin, 2):  # verify
                new_fitness = 0
                test_ok = False
        new_solution.perf = new_fitness  # setter

        if new_fitness < 0:
            # once illegal memory access, subsequent call may fail
            # so we early stop here
            logger.warning(
                f'early stop at iter: {cnt}, current_fitness: {current_fitness:.2f}, new_fitness: {new_fitness:.2f}, best_fitness: {best_fitness:.2f}; temperature: {temperature:.2f}'
            )
            break

        logger.info(
            f'iter: {cnt}, current_fitness: {current_fitness:.2f}, new_fitness: {new_fitness:.2f}, best_fitness: {best_fitness:.2f}; temperature: {temperature:.2f}'
        )
        if test_ok and acceptance_probability(current_fitness, new_fitness,
                                              temperature,
                                              noise_factor) > random.random():
            current_solution = new_solution
            current_fitness = new_fitness

        temperature *= 1 - cooling_rate
        cnt += 1

        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_solution = current_solution

    return best_solution, best_fitness


def run_simulated_annealing(
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

    # sa config
    n_choices=1,
    max_iterations=5000,
    temperature=1.0,
    cooling_rate=0.003,
    policy='single',
    noise_factor=0.0,

    # other config
    seed=0,
    total_flops=None,
    save_suffix='',
    save_dir=None,
    warmup=100,
    rep=100,
):
    logger.info(
        'run simulated annealing with n_choices %d; max_iterations %d; temperature %f; cooling_rate %f; policy %s; noise_factor %f ',
        n_choices, max_iterations, temperature, cooling_rate, policy,
        noise_factor)

    # print(f'bin id {id(bin)}')

    # get initial cubin and asm (the initial have to file IO)
    with tempfile.NamedTemporaryFile(mode='wb', delete=True) as temp_file:

        cubin = bin.asm['cubin']
        temp_file.write(cubin)
        # Ensure data is written to the file before reading it
        temp_file.flush()
        temp_file.seek(0)

        time.sleep(1)
        cf = CubinFile(temp_file.name)

    # ===== config =====
    config = {
        'atol': 1e-2,
        "total_flops": total_flops,
        'warmup': warmup,
        'rep': rep,
    }

    eng = MutationEngine(
        bin,
        cf,
        config,
        grid_0,
        grid_1,
        grid_2,
        stream,
        launch_enter_hook,
        launch_exit_hook,
        non_constexpr_arg_values,
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

    # ===== start =====
    initial_solution = SimulatedSample(eng.kernel_section, eng)
    _ = initial_solution.get_mutable()
    init_perf = max([eng.get_init_perf() for _ in range(5)])
    logger.info(f'init perf: {init_perf:.2f}; dims: {initial_solution.dims}')
    if init_perf < 0:
        raise RuntimeError(f'init perf {init_perf} < 0; not valid cubin')

    _t1 = time.perf_counter()
    best_solution, best_perf = simulated_annealing(
        initial_solution,
        init_perf,
        n_choices,
        max_iterations,
        temperature,
        cooling_rate,
        policy,
        noise_factor,
        eng,
        test_fn,
    )

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
        seed,
        save_suffix,
        save_dir,
        algo='sa',
    )
    # print(f'final bin id {id(bin)}')
    return path