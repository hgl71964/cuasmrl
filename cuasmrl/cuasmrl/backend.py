import os
import sys
import subprocess
import tempfile
import time
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
from cuasmrl.bench import do_bench

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

        self.profile = False
        if os.getenv("SIP_PROFILE", "0") == "1":
            self.profile = True

        # spaces
        sample = Sample(self.eng.kernel_section, self.eng)
        dims, total, mem_loc, max_src_len = sample.get_mutable()
        logger.info(
            f'[INIT] dims: {dims}; total kernel lineno: {total}; mem_loc: {len(mem_loc)}; max_src_len: {max_src_len};'
        )
        self.mem_loc = mem_loc
        self.max_src_len = max_src_len  # because src is variable

        # n line, each line can move up or down; total number unchanged throughout rollouts
        # self.action_space = MultiDiscrete([dims, 2])
        self.action_space = Discrete(n=dims * 2)  # flatten multiDsiscrete

        # NOTE: see Sample.embedding() for state space design
        n_feat = 10 + 1 + 1 + 1 + max_src_len
        self.observation_space = Box(
            low=-1.0,
            high=max(16.0, len(mem_loc)),  # max stall count 16
            shape=(1, total, n_feat),
            dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.sample = Sample(self.eng.kernel_section, self.eng)

        # init_perf, _ = max([self.eng.get_init_perf() for _ in range(1)])
        init_perf, _ = self.eng.get_init_perf()
        self.init_perf = init_perf
        self.last_perf = init_perf

        logger.info(f'[RESET] init perf: {init_perf:.2f};')
        if init_perf < 0:
            raise RuntimeError(f'init perf {init_perf} < 0; not valid cubin')

        state, masks = self._build_state()
        return state, {
            'masks': masks,
            'status': Status.OK,
        }

    def step(self, action):
        action = action.flatten()[0]
        index, direction = action // 2, action % 2
        self.sample.apply(index, direction)

        # run and test
        t1 = time.time()
        perf, cubin = self.eng.get_perf(self.sample)
        test_ok = True
        if perf > 0:
            try:
                test_ok = self.eng.test_fn(
                    cubin,
                    self.n_tests,
                    self.n_tests,
                    False,
                )
            except:
                # segfault from test_fn
                perf = -1
        t2 = time.time()
        if self.profile:
            logger.info(f'[GET PERF] {t2-t1:.2f}s')

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
                for i in range(15, 1, -1):
                    logger.error(f'{self.sample.kernel_section[lineno-i]}')

                logger.critical(f'{self.sample.kernel_section[lineno]}')
                logger.critical(f'{self.sample.kernel_section[lineno-1]}')

                for i in range(1, 15):
                    logger.error(f'{self.sample.kernel_section[lineno+i]}')
            else:
                for i in range(15, 0, -1):
                    logger.error(f'{self.sample.kernel_section[lineno-i]}')

                logger.critical(f'{self.sample.kernel_section[lineno+1]}')
                logger.critical(f'{self.sample.kernel_section[lineno]}')

                for i in range(2, 15):
                    logger.error(f'{self.sample.kernel_section[lineno+i]}')
        elif not test_ok:
            # test failed
            info['status'] = Status.TESTFAIL
            reward = -1
            terminated = True

            # trace error
            lineno = self.sample.candidates[index]
            logger.error(f'TESTFAIL: {index}, {lineno}; {direction}')
            if direction == 0:
                # it was pushed up
                for i in range(15, 1, -1):
                    logger.error(f'{self.sample.kernel_section[lineno-i]}')

                logger.critical(f'{self.sample.kernel_section[lineno]}')
                logger.critical(f'{self.sample.kernel_section[lineno-1]}')

                for i in range(1, 15):
                    logger.error(f'{self.sample.kernel_section[lineno+i]}')
            else:
                for i in range(15, 0, -1):
                    logger.error(f'{self.sample.kernel_section[lineno-i]}')

                logger.critical(f'{self.sample.kernel_section[lineno+1]}')
                logger.critical(f'{self.sample.kernel_section[lineno]}')

                for i in range(2, 15):
                    logger.error(f'{self.sample.kernel_section[lineno+i]}')
        else:
            # valid
            info['status'] = Status.OK
            reward = (self.last_perf - perf) / self.init_perf * 100

        # update
        state, masks = self._build_state()
        info['masks'] = masks
        self.last_perf = perf

        return state, reward, terminated, truncated, info

    def _build_state(self):

        if self.profile:
            t1 = time.time()
            state = self.sample.embedding(
                self.observation_space,
                self.mem_loc,
                self.max_src_len,
            )
            t2 = time.time()
            logger.info(f'[BUILD STATE] {t2-t1:.2f}s')
        else:
            state = self.sample.embedding(
                self.observation_space,
                self.mem_loc,
                self.max_src_len,
            )
        return state


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

        # NOTE: if want to dump the initial cuasm file
        # with open('tmp.cuasm', 'w') as f:
        #     f.write(text_buffer_1.getvalue())

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

        self.cap = CuAsmParser()

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

        # post-process src; e.g. ['desc[UR16][R10.64] -> UR16, R10
        processed_src = []
        for i, word in enumerate(src):
            if word.startswith('desc'):
                w = word.replace(']', '').split('[')
                for r in w[1:]:
                    tmp = r.split('.')[0]  # R10.64 -> R10
                    processed_src.append(tmp)
            elif word.startswith('c'):
                processed_src.append(word)
            else:
                tmp = word.strip(']').strip('[')
                tmp = tmp.split('.')[0]  # R10.64 -> R10
                tmp = tmp.split('+')[0]  # R10+0x2000 -> R10
                processed_src.append(tmp)

        # post-process dest; e.g. [R219+0x4000] -> R219
        if dest is not None:
            if dest.startswith('desc'):
                w = dest.replace(']', '').split('[')
                for r in w[1:]:
                    tmp = r.split('.')[0]  # R10.64 -> R10
                    # In this case, it is treated as src
                    # e.g. STG.E desc[UR16][R10.64], R197 ;
                    # the dst needs to be ready
                    processed_src.append(tmp)
            else:
                dest = dest.strip(']').strip('[')
                dest = dest.split('.')[0]
                dest = dest.split('+')[0]

        return ctrl_code, comment, predicate, opcode, dest, processed_src

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
                # ms = triton.testing.do_bench(fn, warmup=100, rep=100)
                ms = do_bench(fn, 100, 100)
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

        return -ms, None

    # @lru_cache(maxsize=1000)
    def get_perf(self, sample: Sample):
        mutated_kernel = sample.kernel_section[self.kernel_start_line:]
        mutated_sass = deepcopy(self.sass)
        mutated_sass[self.start_line:self.end_line + 1] = mutated_kernel

        # buffer IO
        # cap = CuAsmParser()
        assemble_ok = True
        cubin = None
        try:
            # cap.parse_from_buffer(mutated_sass)
            # cubin = cap.dump_cubin()
            self.cap.parse_from_buffer(mutated_sass)
            cubin = self.cap.dump_cubin()
            self.update_cubin(cubin)
        except Exception as e:
            print(f'Assemble failed: {e}')
            assemble_ok = False
            cubin = None

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
                # ms = triton.testing.do_bench(fn, warmup=100, rep=100)
                ms = do_bench(fn, 100, 100)
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
                # warmup = self.config['warmup']
                # rep = self.config['rep']
                # ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
                ms = do_bench(fn, 100, 100)
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
            return tflops

        return -ms
