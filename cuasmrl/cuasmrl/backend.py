import sys
import subprocess
import tempfile
from copy import deepcopy
from functools import lru_cache

import numpy as np

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.spaces import Box, MultiDiscrete, Discrete, Text

import triton
from triton.compiler.compiler import CompiledKernel

from CuAsm.CuAsmParser import CuAsmParser

from cuasmrl.sample import Sample
from cuasmrl.utils.logger import get_logger
from cuasmrl.utils.constants import Status

logger = get_logger(__name__)

register(
    id="cuasmenv-v0",
    entry_point="cuasmrl.backend:Env",
)


def make_env(
    env_id,
    eng,
    config,
):

    def thunk():
        env = gym.make(env_id,
                       eng=eng,
                       n_tests=config.n_tests,
                       verbose=bool(config.verbose))

        # utility wrapper
        if config.horizon is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=config.horizon)
        if bool(config.normalize_reward):
            env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # seed env
        env.observation_space.seed(config.seed)
        env.action_space.seed(config.seed)
        return env

    return thunk


class Env(gym.Env):

    def __init__(self, eng, n_tests, verbose):
        super().__init__()
        self.eng = eng
        self.n_tests = n_tests
        self.verbose = verbose

        # spaces
        sample = Sample(self.eng.kernel_section, self.eng)
        dims, total = sample.get_mutable()
        logger.info(f'[INIT] dims: {dims}; total: {total};')

        # n line, each line can move up or down; total number unchanged throughout
        # self.action_space = MultiDiscrete([dims, 2])
        self.action_space = Discrete(n=dims * 2)  # flatten multiDsiscrete

        # see Sample.embedding() for state space design
        n_feat = 6
        self.observation_space = Box(
            low=0.0,
            high=16.0,  # max stall count == 16
            shape=(1, total, n_feat),
            dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.sample = Sample(self.eng.kernel_section, self.eng)

        init_perf, _ = max([self.eng.get_init_perf() for _ in range(1)])
        self.init_perf = init_perf
        self.last_perf = init_perf

        logger.info(f'[RESET] init perf: {init_perf:.2f};')
        if init_perf < 0:
            raise RuntimeError(f'init perf {init_perf} < 0; not valid cubin')

        state, masks = self._build_state()
        return state, {'masks': masks}

    def step(self, action):
        action = action.flatten()[0]
        index, direction = action // 2, action % 2
        self.sample.apply(index, direction)

        # run and test
        perf, cubin = self.eng.get_perf(self.sample)
        test_ok = True
        if perf > 0:
            # if not self.eng.test_fn(cubin, self.n_tests, self.n_tests,
            #                         self.verbose):
            #     test_ok = False
            try:
                ok = self.eng.test_fn(cubin, self.n_tests, self.n_tests,
                                      self.verbose)
                test_ok = ok
            except:
                # segfault from test_fn
                perf = -1

        truncated = False
        terminated = False
        info = {}
        reward = None
        state = None
        if perf < 0:
            # segfault
            info['status'] = Status.SEGFAULT
            reward = -1

            # trace segfault
            lineno = self.sample.candidates[index]
            logger.error(f'SEGFAULT: {index}, {lineno}; {direction}')
            if direction == 0:
                # it was pushed up
                logger.error(f'{self.sample.kernel_section[lineno-5]}')
                logger.error(f'{self.sample.kernel_section[lineno-4]}')
                logger.error(f'{self.sample.kernel_section[lineno-3]}')
                logger.error(f'{self.sample.kernel_section[lineno-2]}')
                logger.error(f'{self.sample.kernel_section[lineno]}')

                logger.critical(f'{self.sample.kernel_section[lineno-1]}')

                logger.error(f'{self.sample.kernel_section[lineno+1]}')
                logger.error(f'{self.sample.kernel_section[lineno+2]}')
                logger.error(f'{self.sample.kernel_section[lineno+3]}')
                logger.error(f'{self.sample.kernel_section[lineno+4]}')
                logger.error(f'{self.sample.kernel_section[lineno+5]}')
            else:
                logger.error(f'{self.sample.kernel_section[lineno+1]}')
                logger.error(f'{self.sample.kernel_section[lineno]}')
        elif not test_ok:
            # test failed
            info['status'] = Status.TESTFAIL
            reward = -1
            terminated = True
        else:
            # valid
            info['status'] = Status.OK
            reward = (self.last_perf - perf) / self.init_perf

        # update
        state, masks = self._build_state()
        info['masks'] = masks
        self.last_perf = perf

        return state, reward, terminated, truncated, info

    def _build_state(self):
        return self.sample.embedding(self.observation_space)


class MutationEngine:

    def __init__(
        self,
        bin: CompiledKernel,
        cf,
        grid_0,
        grid_1,
        grid_2,
        stream,
        launch_enter_hook,
        launch_exit_hook,
        non_constexpr_arg_values,
        total_flops,
    ):
        self.bin = bin

        # get sass
        text_buffer_1, text_buffer_2 = cf.dump_sass()
        sass = text_buffer_1.getvalue().split('\n')
        kernel_section = text_buffer_2.getvalue().split('\n')

        # in-memory sass text TODO assme only one kernel
        kernel_label = None
        kernel_start_line = None
        for i, line in enumerate(kernel_section):
            if '.text.' in line:
                kernel_label = line
                kernel_start_line = i
        assert kernel_label is not None, f'Could not find kernel label'

        start_line = None
        for i, line in enumerate(sass):
            if kernel_label == line:
                start_line = i
                break
        assert start_line is not None, f'Could not find start line'

        end_line = None
        line = start_line
        k_line = kernel_start_line
        while line < len(sass) and k_line < len(kernel_section):
            if kernel_section[k_line] != sass[line]:
                end_line = line
                break
            k_line += 1
            line += 1

        if end_line is None:
            assert sass[line - 1] == kernel_section[
                k_line - 1], f'{sass[end_line]} vs {kernel_section[k_line-1]}'
            end_line = line - 1

        self.start_line = start_line
        self.kernel_start_line = kernel_start_line
        self.end_line = end_line
        self.sass = sass
        self.kernel_section = kernel_section

        self.cf = cf

        self.total_flops = total_flops

        self.grid_0 = grid_0
        self.grid_1 = grid_1
        self.grid_2 = grid_2
        self.stream = stream
        self.launch_enter_hook = launch_enter_hook
        self.launch_exit_hook = launch_exit_hook
        self.non_constexpr_arg_values = non_constexpr_arg_values

    def decode(self, line: str):
        line = line.strip('\n')
        line = line.split(' ')
        n = len(line)

        ctrl_code = None
        predicate = None
        comment = None
        opcode = None
        dest = None
        src = []

        # ctrl
        idx = -1
        for i in range(0, n):
            if line[i] != '':
                idx = i
                ctrl_code = line[i]
                break
        assert idx > -1, f'no ctrl: {line}'

        if ctrl_code.startswith('.'):
            # labels
            return None, None, None, None, None, None

        # comment
        for i in range(idx + 1, n):
            if line[i] != '':
                idx = i
                comment = line[i]
                break

        # predicate
        for i in range(idx + 1, n):
            if line[i] != '':

                if line[i][0] == '@':
                    predicate = line[i]
                else:
                    opcode = line[i]

                idx = i
                break

        # opcode
        if opcode is None:
            for i in range(idx + 1, n):
                if line[i] != '':
                    opcode = line[i]
                    idx = i
                    break

        # operand
        for i in range(idx + 1, n):
            if line[i] != '':
                dest = line[i].strip(',')
                idx = i
                break

        if dest == ';':
            # LDGDEPBAR inst
            dest = None

        for i in range(idx + 1, n):
            if line[i] == ';':
                break

            if line[i] != '':
                src.append(line[i].strip(','))

        return ctrl_code, comment, predicate, opcode, dest, src

    def decode_ctrl_code(self, ctrl_code: str):
        ctrl_code = ctrl_code.split(':')
        assert len(ctrl_code) == 5, f'invalid ctrl code: {ctrl_code}'

        barr = ctrl_code[0][2:]
        waits = []
        for bar in barr:
            if bar != '-':
                waits.append(int(bar))

        read = ctrl_code[1]
        write = ctrl_code[2]
        yield_flag = ctrl_code[3]
        stall_count = ctrl_code[4]
        return waits, read, write, yield_flag, stall_count

    def update_cubin(self, cubin):
        self.bin.asm['cubin'] = cubin
        self.bin.cu_module = None  # force to re-load

    def get_init_perf(self):
        mutated_sass = self.sass

        # buffer IO
        cap = CuAsmParser()
        assemble_ok = True
        try:
            cap.parse_from_buffer(mutated_sass)
            cubin = cap.dump_cubin()
            self.update_cubin(cubin)
        except Exception as e:
            print(f'Assemble failed: {e}')
            assemble_ok = False

        # BENCH
        fn = lambda: self.bin.c_wrapper(
            self.grid_0,
            self.grid_1,
            self.grid_2,
            self.bin.num_warps,
            self.bin.num_ctas,
            self.bin.clusterDims[0],
            self.bin.clusterDims[1],
            self.bin.clusterDims[2],
            self.bin.shared,
            self.stream,
            self.bin.cu_function,
            self.launch_enter_hook,
            self.launch_exit_hook,
            self.bin,
            *self.bin.assemble_tensormap_to_arg(self.non_constexpr_arg_values),
        )
        if assemble_ok:
            try:
                ms = triton.testing.do_bench(fn, warmup=100, rep=100)
            except RuntimeError as run_err:
                # likely a cuda error
                print(f'CUDA? Runtime Err: {run_err}')
                ms = -1
            except Exception as e:
                print(f'Other error: {e}')
                raise e
        else:
            ms = -1

        if self.total_flops is not None:
            tflops = self.total_flops / ms * 1e-9
            # print(f'ms: {ms:.3f}; tflops: {tflops:.3f};')
            return tflops, cubin

        return -ms

    @lru_cache(maxsize=1000)
    def get_perf(self, sample: Sample):
        mutated_kernel = sample.kernel_section[self.kernel_start_line:]
        mutated_sass = deepcopy(self.sass)
        mutated_sass[self.start_line:self.end_line + 1] = mutated_kernel

        # buffer IO
        cap = CuAsmParser()
        assemble_ok = True
        cubin = None
        try:
            cap.parse_from_buffer(mutated_sass)
            cubin = cap.dump_cubin()
            self.update_cubin(cubin)
        except Exception as e:
            print(f'Assemble failed: {e}')
            assemble_ok = False
            cubin = None

        ## XXX NOT test here to allow possible intermediate incorrect results
        # BENCH
        fn = lambda: self.bin.c_wrapper(
            self.grid_0,
            self.grid_1,
            self.grid_2,
            self.bin.num_warps,
            self.bin.num_ctas,
            self.bin.clusterDims[0],
            self.bin.clusterDims[1],
            self.bin.clusterDims[2],
            self.bin.shared,
            self.stream,
            self.bin.cu_function,
            self.launch_enter_hook,
            self.launch_exit_hook,
            self.bin,
            *self.bin.assemble_tensormap_to_arg(self.non_constexpr_arg_values),
        )
        if assemble_ok:
            try:
                ms = triton.testing.do_bench(fn, warmup=100, rep=100)
            except RuntimeError as run_err:
                # likely a cuda error
                logger.error(f'CUDA? Runtime Err: {run_err}')
                ms = -1
                cubin = None
            except Exception as e:
                logger.error(f'Other error: {e}')
                raise e
        else:
            ms = -1

        if self.total_flops is not None:
            tflops = self.total_flops / ms * 1e-9
            # print(f'ms: {ms:.3f}; tflops: {tflops:.3f};')
            return tflops, cubin

        # print(f'ms: {ms:.3f};')
        raise NotImplementedError()

    def assemble(self, sample: Sample):
        mutated_kernel = sample.kernel_section[self.kernel_start_line:]
        mutated_sass = deepcopy(self.sass)
        mutated_sass[self.start_line:self.end_line + 1] = mutated_kernel

        # buffer IO
        cap = CuAsmParser()
        assemble_ok = True
        try:
            cap.parse_from_buffer(mutated_sass)
            cubin = cap.dump_cubin()
            self.update_cubin(cubin)  # in place update
        except Exception as e:
            print(f'Assemble failed: {e}')
            assemble_ok = False
            raise e

        # final BENCH
        fn = lambda: self.bin.c_wrapper(
            self.grid_0,
            self.grid_1,
            self.grid_2,
            self.bin.num_warps,
            self.bin.num_ctas,
            self.bin.clusterDims[0],
            self.bin.clusterDims[1],
            self.bin.clusterDims[2],
            self.bin.shared,
            self.stream,
            self.bin.cu_function,
            self.launch_enter_hook,
            self.launch_exit_hook,
            self.bin,
            *self.bin.assemble_tensormap_to_arg(self.non_constexpr_arg_values),
        )
        if assemble_ok:
            try:
                warmup = self.config['warmup']
                rep = self.config['rep']
                ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            except RuntimeError as run_err:
                # likely a cuda error
                print(f'CUDA? Runtime Err: {run_err}')
                ms = -1
            except Exception as e:
                print(f'Other error: {e}')
                raise e
        else:
            ms = -1

        total_flops = self.config.get('total_flops', None)

        if total_flops is not None:
            tflops = total_flops / ms * 1e-9
            sample.perf = tflops
            # print(f'ms: {ms:.3f}; tflops: {tflops:.3f};')
            return tflops

        return -ms
