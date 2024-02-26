from collections import defaultdict, namedtuple

from triton.runtime.jit import T, JITFunction, KernelArg, get_current_device, set_current_device, get_cuda_stream
from triton.compiler.compiler import CompiledKernel, compile, get_arch_default_num_stages, get_arch_default_num_warps
from triton.common.backend import get_backend, get_cuda_version_key

from cuasmrl.simulated_annealing import run_simulated_annealing
from cuasmrl.selection import run_selection

from cuasmrl.compiler import compile as fgk_compile
from cuasmrl.compiler import CompiledKernel as fgk_CompiledKernel


def jit(
    fn,
    *,
    version=None,
    do_not_specialize=None,
    debug=None,
    noinline=None,
):

    def decorator(fn: T) -> JITFunction[T]:
        assert callable(fn)
        return ASMJITFunction(
            fn,
            version=None,
            do_not_specialize=None,
            debug=None,
            noinline=None,
        )

    if fn is not None:
        return decorator(fn)
    else:
        return decorator


class ASMJITFunction(JITFunction):

    def __init__(
        self,
        fn,
        version=None,
        do_not_specialize=None,
        debug=None,
        noinline=None,
    ):
        super().__init__(fn, version, do_not_specialize, debug, noinline)

        self.search_cache = defaultdict(dict)

        self.max_iterations = 1000
        self.temperature = 0.4
        self.cooling_rate = 0.003
        self.policy = 'single'
        self.noise_factor = 0.0
        self.n_test_samples = 100
        self.seed = 0
        self.save_dir = 'tmp'
        self.total_flops = 1e9
        self.save_suffix = ''

    def search(self, *args, **kwargs):

        # Get a compiler-flags arg like `num_warps` and remove it from kwargs.
        def get_special_arg(name: str, default=None):
            if name not in kwargs:
                return default
            ret = kwargs[name]
            del kwargs[name]
            return ret

        grid = get_special_arg("grid")
        num_warps = get_special_arg("num_warps")
        num_ctas = get_special_arg("num_ctas", 1)
        num_stages = get_special_arg("num_stages")
        enable_warp_specialization = get_special_arg(
            "enable_warp_specialization", False)
        enable_fp_fusion = get_special_arg("enable_fp_fusion", True)
        extern_libs = get_special_arg("extern_libs")
        stream = get_special_arg("stream")
        warmup = get_special_arg("warmup", False)
        device = get_special_arg("device")
        device_type = get_special_arg("device_type")

        # gh512 test args
        ret_ptr = get_special_arg("ret_ptr")
        test_inputs = get_special_arg("test_inputs")
        test_outputs = get_special_arg("test_outputs")
        load_dir = get_special_arg("load_dir")

        # Bind the remaining arguments to `fn`.
        bound_args = self.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        assert len(bound_args.arguments) == len(self.params)
        args = [
            KernelArg(arg_value, param)
            for (_, arg_value
                 ), param in zip(bound_args.arguments.items(), self.params)
        ]

        non_constexpr_arg_values = [
            arg.value for arg in args if not arg.param.is_constexpr
        ]

        sig_key = tuple(arg.signature_key() for arg in args
                        if not arg.param.is_constexpr)
        spec_key = tuple(arg.specialization_key() for arg in args
                         if not arg.param.do_not_specialize)
        constexpr_key = tuple(arg.value for arg in args
                              if arg.param.is_constexpr)

        assert num_ctas > 0
        assert grid is not None
        if callable(grid):
            # Arguments are passed as a dict to `grid`, by contract.
            # TODO(jlebar): In the new launch API, pass the compiler flags as a
            # second parameter to `grid`.
            grid = grid(dict(bound_args.arguments))
        grid_size = len(grid)
        grid_0 = grid[0]
        grid_1 = grid[1] if grid_size > 1 else 1
        grid_2 = grid[2] if grid_size > 2 else 1
        if device_type is None:
            device_types = [
                self._device_of(arg) for arg in non_constexpr_arg_values
            ]
            device_types = [
                _device_type for _device_type in device_types
                if _device_type != ""
            ]
            device_type = self._conclude_device_type(device_types, [
                self._pinned_memory_of(arg) for arg in non_constexpr_arg_values
            ])

        device_backend = None
        if device_type not in ["cuda"]:
            device_backend = get_backend(device_type)
            if device_backend is None:
                raise ValueError("Cannot find backend for " + device_type)

        if device is None:
            if device_type in ["cuda"]:
                device = get_current_device()
                set_current_device(device)
            else:
                device = device_backend.get_current_device()
                device_backend.set_current_device(device)
        if stream is None and not warmup:
            if device_type in ["cuda"]:
                stream = get_cuda_stream(device)
            else:
                stream = device_backend.get_stream()

        if num_warps is None:
            num_warps = get_arch_default_num_warps(device_type)
        if num_stages is None:
            num_stages = get_arch_default_num_stages(device_type)

        if device_type in ["cuda"]:
            version_key = get_cuda_version_key()
        else:
            version_key = device_backend.get_version_key()
        key = (
            version_key,
            sig_key,
            constexpr_key,
            spec_key,
            num_warps,
            num_ctas,
            num_stages,
            enable_warp_specialization,
            enable_fp_fusion,
            self.debug,
        )
        if extern_libs is not None:
            key = (key, tuple(extern_libs.items()))

        # Kernel is not cached; we have to compile.
        if key not in self.search_cache[device]:
            configs = (self._get_config(*[arg.value for arg in args]), )
            constants = {
                arg.param.num: arg.value
                for arg in args if arg.param.is_constexpr
                or arg.param.num in configs[0].equal_to_1 or arg.value is None
            }
            for i, arg in constants.items():
                if callable(arg):
                    raise TypeError(
                        f"Callable constexpr at index {i} is not supported")

            # Build kernel signature -- doesn't include constexpr arguments.
            signature = {
                arg.param.num: self._type_of(self._key_of(arg.value))
                for arg in args if not arg.param.is_constexpr
            }

            if self._call_hook(
                    key,
                    signature,
                    device,
                    constants,
                    num_warps,
                    num_ctas,
                    num_stages,
                    enable_warp_specialization,
                    enable_fp_fusion,
                    extern_libs,
                    configs,
            ):
                return None

            so_path, metadata, asm = fgk_compile(
                self,
                signature=signature,
                device=device,
                constants=constants,
                num_warps=num_warps,
                num_ctas=num_ctas,
                num_stages=num_stages,
                enable_warp_specialization=enable_warp_specialization,
                enable_fp_fusion=enable_fp_fusion,
                extern_libs=extern_libs,
                configs=configs,
                debug=self.debug,
                device_type=device_type,
            )
            if load_dir is None:
                bin = fgk_CompiledKernel(so_path, metadata, asm)
                run_simulated_annealing(
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
                    stream,  #
                    CompiledKernel.launch_enter_hook,
                    CompiledKernel.launch_exit_hook,  #
                    # algo
                    1,
                    self.max_iterations,
                    self.temperature,
                    self.cooling_rate,
                    self.policy,
                    self.noise_factor,
                    seed=self.seed,
                    total_flops=self.total_flops,
                    save_suffix=self.save_suffix,
                    save_dir=self.save_dir,
                    warmup=100,
                    rep=100,
                )
            else:
                bin = run_selection(
                    so_path,
                    metadata,
                    asm,
                    args,
                    sig_key,
                    non_constexpr_arg_values,
                    ret_ptr,
                    test_inputs,
                    test_outputs,
                    grid_0,
                    grid_1,
                    grid_2,
                    stream,  #
                    CompiledKernel.launch_enter_hook,
                    CompiledKernel.launch_exit_hook,  # 
                    cubin_dir_path=load_dir,
                    n_test_samples=self.n_test_samples,
                )
            warn = '\033[93m'
            end = '\033[0m'
            print(f"{warn}SIP JIT{end}")
            self.search_cache[device][key] = bin

        bin = self.search_cache[device][key]
        if not warmup:
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
                CompiledKernel.launch_enter_hook,
                CompiledKernel.launch_exit_hook,
                bin,
                *bin.assemble_tensormap_to_arg(non_constexpr_arg_values),
            )
        return bin

    def run(self, *args, **kwargs):
        return self.search(*args, **kwargs)

    # execute triton's default run
    def triton_run(self, *args, **kwargs):
        return super().run(*args, **kwargs)