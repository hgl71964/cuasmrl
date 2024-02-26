import sys
import subprocess
from multiprocessing import Process, Queue

import tempfile
from functools import lru_cache

from copy import deepcopy

import triton
from triton.compiler.compiler import CompiledKernel

from CuAsm.CuAsmParser import CuAsmParser

from cuasmrl.sample import Sample
from cuasmrl.utils.logger import get_logger

logger = get_logger(__name__)


class MutationEngine:

    def __init__(
        self,
        bin: CompiledKernel,
        cf,
        config: dict,
        grid_0,
        grid_1,
        grid_2,
        stream,
        launch_enter_hook,
        launch_exit_hook,
        non_constexpr_arg_values,
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

        self.config = config

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

        idx = -1
        for i in range(0, n):
            if line[i] != '':
                idx = i
                ctrl_code = line[i]
                break
        assert idx > -1, f'no ctrl: {line}'

        for i in range(idx + 1, n):
            if line[i] != '':
                idx = i
                comment = line[i]
                break

        for i in range(idx + 1, n):
            if line[i] != '':

                if line[i][0] == '@':
                    predicate = line[i]
                else:
                    opcode = line[i]

                idx = i
                break

        if opcode is None:
            for i in range(idx + 1, n):
                if line[i] != '':
                    opcode = line[i]
                    idx = i
                    break

        for i in range(idx + 1, n):
            if line[i] != '':
                dest = line[i].strip(',')
                idx = i
                break

        for i in range(idx + 1, n):
            if line[i] == ';':
                break

            if line[i] != '':
                src.append(line[i])

        return ctrl_code, comment, predicate, opcode, dest, src

    def decode_ctrl_code(self, ctrl_code: str):
        ctrl_code = ctrl_code.split(':')
        assert len(ctrl_code) == 5, f'invalid ctrl code: {ctrl_code}'

        barr = ctrl_code[0]
        read = ctrl_code[1]
        write = ctrl_code[2]
        yield_flag = ctrl_code[3]
        stall_count = ctrl_code[4]
        return barr, read, write, yield_flag, stall_count

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
            # print(f'ms: {ms:.3f}; tflops: {tflops:.3f};')
            return tflops

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
                warmup = self.config['warmup']
                rep = self.config['rep']
                ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
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

        total_flops = self.config.get('total_flops', None)

        if total_flops is not None:
            tflops = total_flops / ms * 1e-9
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
