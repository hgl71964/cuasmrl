import os
import pickle
import time
import builtins
from typing import Dict, Union, Optional
from copy import deepcopy
from collections import defaultdict, namedtuple

from triton.testing import do_bench
from triton.runtime.autotuner import Autotuner as TritonAutotuner
from triton.runtime.autotuner import OutOfResources

from cuasmrl.jit import asm_JITFunction
from cuasmrl.utils.logger import get_logger
from cuasmrl.utils.gpu_utils import get_gpu_name

logger = get_logger(__name__)


class Autotuner(TritonAutotuner):
    # 1. we use Triton's Autotuner to search for good kernel configurations
    # 2. then search for good assembly schedule

    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        prune_configs_by,
        warmup,
        rep,

        # gh512
        ret_ptr,
        **kwargs,
    ):
        super().__init__(
            fn,
            arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            prune_configs_by,
            warmup,
            rep,
        )

        self.cache_config = None
        self.ret_ptr = ret_ptr

        # workload
        self.total_flops = kwargs.get('total_flops', 1e9)
        self.seed = kwargs.get('seed', 0)
        self.save_suffix = kwargs.get('save_suffix', '')
        self.save_dir = kwargs.get('save_dir', None)
        self.n_test_samples = kwargs.get('n_test_samples', 100)

        # sa
        self.sa_runs = kwargs.get('sa_runs', 10)
        self.max_iterations = kwargs.get('max_iterations', 1000)
        self.temperature = kwargs.get('temperature', 0.4)
        self.cooling_rate = kwargs.get('cooling_rate', 0.003)
        self.noise_factor = kwargs.get('noise_factor', 0.0)
        self.policy = kwargs.get('policy', 'single')
        # ga
        self.population_size = kwargs.get('population_size', 100)
        self.generations = kwargs.get('generations', 50)
        self.mutation_rate = kwargs.get('mutation_rate', 0.1)
        self.tournament_size = kwargs.get('tournament_size', 5)

        # at this time, fn has been init, so we overwrite the default args
        self.fn.total_flops = self.total_flops
        self.fn.seed = self.seed
        self.fn.save_suffix = self.save_suffix
        self.fn.save_dir = self.save_dir
        self.fn.n_test_samples = self.n_test_samples

        self.fn.sa_runs = self.sa_runs
        self.fn.max_iterations = self.max_iterations
        self.fn.temperature = self.temperature
        self.fn.cooling_rate = self.cooling_rate
        self.fn.noise_factor = self.noise_factor
        self.fn.policy = self.policy

        self.fn.population_size = self.population_size
        self.fn.generations = self.generations
        self.fn.mutation_rate = self.mutation_rate
        self.fn.tournament_size = self.tournament_size

    def _bench(self, *args, config, **meta):
        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(
                f"Conflicting meta-parameters: {', '.join(conflicts)}."
                " Make sure that you don't re-define auto-tuned symbols.")
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.kwargs)
        full_nargs = {**self.nargs, **current}

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.pre_hook(args)

            # gh512
            self.fn.triton_run(
                *args,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
                num_ctas=config.num_ctas,
                enable_warp_specialization=config.enable_warp_specialization,
                # enable_persistent=False,
                **current,
            )
            self.post_hook(args)

        try:
            # this populates data to the ret_ptr
            return do_bench(kernel_call,
                            warmup=self.warmup,
                            rep=self.rep,
                            quantiles=(0.5, 0.2, 0.8))
        except OutOfResources:
            return [float("inf"), float("inf"), float("inf")]

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))

        def get_special_arg(name: str, default=None):
            if name not in kwargs:
                return default
            ret = kwargs[name]
            del kwargs[name]
            return ret

        ret_ptr = get_special_arg("ret_ptr")
        if self.ret_ptr is not None:
            ret_ptr = self.ret_ptr
        assert ret_ptr is not None, "ret_ptr must be provided"
        # test_inputs = get_special_arg("test_inputs")
        # test_outputs = get_special_arg("test_outputs")
        load_dir = get_special_arg("load_dir")

        if self.cache_config is not None:
            # cached
            config = self.cache_config
        elif len(self.configs) == 1:
            # heuristic
            config = self.configs[0]
        elif self._exist_config():
            # file IO, NOTE in benchmark, don't do it
            config = self.cache_config
        elif len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
            _args = []
            for name in self.arg_names:
                if name in all_args:
                    _args.append(all_args[name])
            key = [_args[i] for i in self.key_idx]
            for arg in _args:
                if hasattr(arg, "dtype"):
                    key.append(str(arg.dtype))
            key = tuple(key)
            if key not in self.cache:
                # prune configs
                pruned_configs = self.prune_configs(kwargs)
                bench_start = time.time()
                timings = {}
                for config in pruned_configs:
                    res = self._bench(*args, config=config, **kwargs)
                    timings[config] = res
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                self.cache[key] = builtins.min(timings, key=timings.get)
                self.pre_hook(args, reset_only=True)
                self.configs_timings = timings
            config = self.cache[key]
            self._write_config(config)
        else:
            raise RuntimeError("No config found")
        self.best_config = config
        full_nargs = {**self.nargs, **kwargs, **self.best_config.kwargs}
        if config.pre_hook is not None:
            config.pre_hook(full_nargs)

        ret = self.fn.search(
            *args,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
            num_ctas=config.num_ctas,
            enable_warp_specialization=config.enable_warp_specialization,
            # gh512
            ret_ptr=ret_ptr,
            # test_inputs=test_inputs,
            # test_outputs=test_outputs,
            load_dir=load_dir,
            **kwargs,
            **config.kwargs,
        )
        self.nargs = None
        return ret

    def _write_config(self, config):
        gpu_name = get_gpu_name()
        dir_path = f'data/{gpu_name}'
        if self.save_dir is not None:
            dir_path = f'data/{gpu_name}/{self.save_dir}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        cache_path = f'{dir_path}/cache_config.pkl'
        with open(cache_path, 'wb') as f:
            pickle.dump(config, f)

    def _exist_config(self) -> bool:
        gpu_name = get_gpu_name()
        dir_path = f'data/{gpu_name}'
        if self.save_dir is not None:
            dir_path = f'data/{gpu_name}/{self.save_dir}'
        if not os.path.exists(dir_path):
            return False

        cache_path = f'{dir_path}/cache_config.pkl'
        if os.path.isfile(cache_path):
            with open(cache_path, 'rb') as f:
                self.cache_config = pickle.load(f)
            return True
        return False


def autotune(
    configs,
    key,

    # the index to the ret_ptr
    ret_ptr: Optional[int],

    # other default
    prune_configs_by=None,
    reset_to_zero=None,
    restore_value=None,
    warmup=100,
    rep=100,
    **kwargs,
):

    def decorator(fn):
        return Autotuner(
            fn,
            fn.arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            prune_configs_by,
            warmup,
            rep,
            ret_ptr,
            **kwargs,
        )

    return decorator
