from __future__ import annotations

import functools
import hashlib
import json
import os
import re
from collections import namedtuple
from pathlib import Path
from typing import Any

from dataclasses import dataclass

from triton._C.libtriton.triton import (ClusterInfo, TMAInfos,
                                        add_external_libs,
                                        compile_ptx_to_cubin, get_env_vars,
                                        get_num_warps, get_shared_memory_size,
                                        ir, runtime, translate_llvmir_to_ptx,
                                        translate_triton_gpu_to_llvmir)
from triton.common.backend import get_backend, get_cuda_version_key, path_to_ptxas
from triton.common.build import is_hip
# from ..runtime import driver, jit, JITFunction
# TODO: runtime.errors
from triton.runtime.autotuner import OutOfResources
from triton.runtime.cache import get_cache_manager, get_dump_manager, get_override_manager
from triton.runtime.driver import driver
from triton.runtime.jit import (JITFunction, get_cuda_stream,
                                get_current_device, get_device_capability)
from triton.tools.disasm import get_sass
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.make_launcher import make_stub
from triton.compiler.utils import (InfoFromBackendForTensorMap,
                                   TensorMapManager, get_ids_of_tensormaps,
                                   parse_tma_info)

from triton.compiler.compiler import (
    # utils
    LazyDict,
    get_arch_default_num_warps,
    get_arch_default_num_stages,
    CudaTargetDescriptor,
    get_cuda_capability,
    get_kernel_name,
    make_hash,
    arg_type_pattern,
    convert_type_repr,
    ttgir_num_warps_pattern,
    _get_jsonable_constants,

    # pipeline
    parse_mlir_module,
    optimize_ttir,
    optimize_ttgir,
    ttir_to_ttgir,
    ttgir_to_llir,
    add_cuda_stages,
    instance_descriptor,
    prototype_pattern,
)


def compile(fn, **kwargs):
    # Get device type to decide which backend should be used
    device_type = kwargs.get("device_type", "cuda")
    capability = kwargs.get("cc", None)

    if is_hip():
        device_type = "hip"
    is_cuda = device_type == "cuda"
    if is_hip():
        is_cuda = False

    context = ir.context()
    constants = kwargs.get("constants", dict())
    num_warps = kwargs.get("num_warps",
                           get_arch_default_num_warps(device_type))
    assert num_warps > 0 and (
        num_warps & (num_warps - 1)) == 0, "num_warps must be a power of 2"
    num_ctas = kwargs.get("num_ctas", 1)
    num_stages = kwargs.get(
        "num_stages",
        get_arch_default_num_stages(device_type, capability=capability))
    enable_fp_fusion = kwargs.get("enable_fp_fusion", True)
    # TODO[shuhaoj]: Default should be to enable warp specialization once possible
    enable_warp_specialization = kwargs.get("enable_warp_specialization",
                                            False)
    # TODO[shuhaoj]: persistent can be decoupled with warp specialization
    enable_persistent = kwargs.get("enable_persistent",
                                   enable_warp_specialization)
    extern_libs = kwargs.get("extern_libs", dict())
    if extern_libs is None:
        extern_libs = dict()
    debug = kwargs.get("debug", False)
    # Flag to control whether to store mma layout directly
    optimize_epilogue = False
    if os.environ.get('OPTIMIZE_EPILOGUE', '') == '1':
        optimize_epilogue = True
    #
    cluster_info = ClusterInfo()
    if "clusterDims" in kwargs:
        cluster_info.clusterDimX = kwargs["clusterDims"][0]
        cluster_info.clusterDimY = kwargs["clusterDims"][1]
        cluster_info.clusterDimZ = kwargs["clusterDims"][2]
    tma_infos = TMAInfos()
    # build architecture descriptor
    if device_type == "cuda":
        _device_backend = get_backend(device_type)
        target = CudaTargetDescriptor(
            capability=get_cuda_capability(capability),
            num_warps=num_warps,
            enable_fp_fusion=enable_fp_fusion)
    else:
        _device_backend = get_backend(device_type)
        assert _device_backend
        target = _device_backend.get_architecture_descriptor(**kwargs)
    # build compilation stages
    stages = dict()
    stages["ast"] = (lambda path: fn, None)
    stages["ttir"] = (lambda path: parse_mlir_module(
        path, context
    ), lambda src: optimize_ttir(
        ast_to_ttir(
            src, signature, configs[0], constants, debug=debug, target=target),
        target))
    if is_cuda:
        stages["ttgir"] = (lambda path: parse_mlir_module(path, context),
                           lambda src: optimize_ttgir(
                               ttir_to_ttgir(src, num_warps, num_ctas, target),
                               num_stages, num_warps, num_ctas, target,
                               cluster_info, enable_warp_specialization,
                               enable_persistent, optimize_epilogue))
        stages["llir"] = (
            lambda path: Path(path).read_text(),
            lambda src: ttgir_to_llir(src, extern_libs, target, tma_infos))
        add_cuda_stages(target, extern_libs, stages)
    elif device_type == "hip":
        _device_backend.add_stages(target,
                                   extern_libs,
                                   stages,
                                   num_warps=num_warps,
                                   num_stages=num_stages)
    else:
        # pass the user's configuration to the backend device.
        target["num_warps"] = num_warps
        target["num_stages"] = num_stages
        target["num_ctas"] = num_ctas
        _device_backend.add_stages(target, extern_libs, stages)

    # find out the signature of the function
    if isinstance(fn, JITFunction):
        configs = kwargs.get("configs", None)
        signature = kwargs["signature"]
        if configs is None:
            configs = [instance_descriptor()]
        assert len(configs) == 1
        kwargs["configs"] = configs
        name = fn.__name__
        first_stage = 0
        if isinstance(signature, str):
            signature = {
                k: v.strip()
                for k, v in enumerate(signature.split(","))
            }
        kwargs["signature"] = signature
    else:
        assert isinstance(fn, str)
        _, ir_name = os.path.basename(fn).split(".")
        src = Path(fn).read_text()
        import re
        match = re.search(prototype_pattern[ir_name], src, re.MULTILINE)
        # TODO: support function attributes at group 3 (e.g., device function)
        name, signature = match.group(1), match.group(2)
        types = re.findall(arg_type_pattern[ir_name], signature)
        if ir_name == 'ttgir':
            num_warps_matches = re.findall(ttgir_num_warps_pattern, src)
            assert len(num_warps_matches
                       ) == 1, "Expected exactly one match for num_warps"
            assert "num_warps" not in kwargs or int(
                num_warps_matches[0]
            ) == num_warps, "num_warps in ttgir does not match num_warps in compile"
            num_warps = int(num_warps_matches[0])
        param_tys = [convert_type_repr(ty) for ty in types]
        signature = {k: v for k, v in enumerate(param_tys)}
        first_stage = list(stages.keys()).index(ir_name)

    # create cache manager
    fn_cache_manager = get_cache_manager(
        make_hash(fn, target, get_env_vars(), _device_backend, **kwargs))
    # managers used to dump and override IR for debugging
    enable_override = os.environ.get("TRITON_KERNEL_OVERRIDE", "0") == "1"
    fn_override_manager = get_override_manager(
        make_hash(fn,
                  target,
                  get_env_vars(),
                  _device_backend,
                  **kwargs,
                  ignore_version=True))
    fn_dump_manager = get_dump_manager(
        make_hash(fn,
                  target,
                  get_env_vars(),
                  _device_backend,
                  **kwargs,
                  ignore_version=True))

    # determine name and extension type of provided function
    if isinstance(fn, JITFunction):
        name, ext = fn.__name__, "ast"
    else:
        name, ext = os.path.basename(fn).split(".")

    # load metadata if any
    metadata = None
    metadata_filename = f"{name}.json"

    # The group is addressed by the metadata
    metadata_group = fn_cache_manager.get_group(metadata_filename) or {}

    metadata_path = metadata_group.get(metadata_filename)

    if metadata_path is not None:
        with open(metadata_path) as f:
            metadata = json.load(f)
            if 'tensormaps_info' in metadata:
                metadata['tensormaps_info'] = [
                    InfoFromBackendForTensorMap(e)
                    for e in metadata['tensormaps_info']
                ]
    else:
        metadata = {
            "num_warps": num_warps,
            "num_ctas": num_ctas,
            "num_stages": num_stages,
            "enable_warp_specialization": enable_warp_specialization,
            "enable_persistent": enable_persistent,
            "constants": _get_jsonable_constants(constants),
            "debug": debug,
            "target": target,
        }
        metadata.update(get_env_vars())
        if ext == "ptx":
            assert "shared" in kwargs, "ptx compilation must provide shared memory size"
            metadata["shared"] = kwargs["shared"]

    # Add device type to meta information
    metadata["device_type"] = device_type

    first_stage = list(stages.keys()).index(ext)
    asm = LazyDict()
    module = fn
    # run compilation pipeline  and populate metadata
    for ir_name, (parse, compile_kernel) in list(stages.items())[first_stage:]:
        ir_filename = f"{name}.{ir_name}"

        if ir_name == ext:
            next_module = parse(fn)
        else:
            path = metadata_group.get(ir_filename)
            if path is None:
                next_module = compile_kernel(module)
                if ir_name == "amdgcn":
                    extra_file_name = f"{name}.hsaco_path"
                    metadata_group[ir_filename] = fn_cache_manager.put(
                        next_module[0], ir_filename)
                    metadata_group[extra_file_name] = fn_cache_manager.put(
                        next_module[1], extra_file_name)
                else:
                    metadata_group[ir_filename] = fn_cache_manager.put(
                        next_module, ir_filename)
                    fn_dump_manager.put(next_module, ir_filename)
                    if (enable_override
                            and fn_override_manager.has_file(ir_filename)):
                        print(f"\nOverriding kernel with file {ir_filename}")
                        full_name = fn_override_manager.get_file(ir_filename)
                        next_module = parse(full_name)
            else:
                if ir_name == "amdgcn":
                    extra_file_name = f"{name}.hsaco_path"
                    hasco_path = metadata_group.get(extra_file_name)
                    assert hasco_path is not None, "Expected to have hsaco_path in metadata when we have the amdgcn"
                    next_module = (parse(path), parse(hasco_path))
                else:
                    next_module = parse(path)

        if ir_name == "cubin":
            asm[ir_name] = next_module
            asm["sass"] = lambda: get_sass(next_module)
        elif ir_name == "amdgcn":
            asm[ir_name] = str(next_module[0])
        else:
            asm[ir_name] = str(next_module)
        if ir_name == "llir" and "shared" not in metadata:
            if is_hip():
                metadata["shared"] = _device_backend.get_shared_memory_size(
                    module)
            else:
                metadata["shared"] = get_shared_memory_size(module)
        if ir_name == "ttgir":
            if is_hip():
                metadata["num_warps"] = _device_backend.get_num_warps(
                    next_module)
            else:
                metadata["enable_warp_specialization"] = ir.is_ws_supported(
                    next_module)
                if metadata["enable_warp_specialization"]:
                    metadata["num_warps"] = get_num_warps(next_module)
        if ir_name == "ptx":
            metadata["name"] = get_kernel_name(next_module,
                                               pattern='// .globl')
        if ir_name == "amdgcn":
            metadata["name"] = get_kernel_name(next_module[0],
                                               pattern='.globl')
            asm["hsaco_path"] = next_module[1]
        if not is_cuda and not is_hip():
            _device_backend.add_meta_info(ir_name, module, next_module,
                                          metadata, asm)
        module = next_module

    ids_of_folded_args = tuple([int(k) for k in configs[0].ids_of_folded_args
                                ]) if isinstance(fn, JITFunction) else ()
    if "clusterDims" not in metadata:
        metadata["clusterDims"] = [
            cluster_info.clusterDimX, cluster_info.clusterDimY,
            cluster_info.clusterDimZ
        ]

    if len(tma_infos) > 0:
        metadata["tensormaps_info"] = parse_tma_info(tma_infos,
                                                     ids_of_folded_args)
    # set constant
    if "tensormaps_info" in metadata:
        for i, _ in enumerate(metadata["tensormaps_info"]):
            metadata["tensormaps_info"][
                i].ids_of_folded_args = ids_of_folded_args

    ids_of_tensormaps = get_ids_of_tensormaps(
        metadata.get("tensormaps_info", None))
    if isinstance(fn, JITFunction) and "tensormaps_info" in metadata:
        fn.tensormaps_info = metadata["tensormaps_info"]

    ids_of_const_exprs = tuple(fn.constexprs) if isinstance(
        fn, JITFunction) else ()
    ids = {
        "ids_of_tensormaps": ids_of_tensormaps,
        "ids_of_folded_args": ids_of_folded_args,
        "ids_of_const_exprs": ids_of_const_exprs
    }
    # cache manager
    if is_cuda:
        so_path = make_stub(
            name,
            signature,
            constants,
            ids,
            enable_warp_specialization=enable_warp_specialization)
    else:
        so_path = _device_backend.make_launcher_stub(name, signature,
                                                     constants, ids)
    # write-back metadata, if it didn't come from the cache
    if metadata_path is None:
        metadata_group[metadata_filename] = fn_cache_manager.put(
            json.dumps(metadata, default=vars),
            metadata_filename,
            binary=False)
    fn_cache_manager.put_group(metadata_filename, metadata_group)

    # gh512: only return the metadata
    return so_path, metadata, asm


class CompiledKernel:
    # just drop the reference to fn

    launch_enter_hook = None
    launch_exit_hook = None
    tensormap_manager = TensorMapManager()

    def __init__(self, so_path, metadata, asm):
        # initialize launcher
        import importlib.util
        spec = importlib.util.spec_from_file_location("__triton_launcher",
                                                      so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.c_wrapper = getattr(mod, "launch")
        # initialize metadata
        self.shared = metadata["shared"]
        self.num_warps = metadata["num_warps"]
        self.num_ctas = metadata["num_ctas"]
        self.num_stages = metadata["num_stages"]
        self.clusterDims = metadata["clusterDims"]
        if "tensormaps_info" in metadata:
            self.tensormaps_info = metadata["tensormaps_info"]
        self.constants = metadata["constants"]
        self.device_type = metadata["device_type"]
        self.device_backend = get_backend(
            self.device_type) if self.device_type not in ["cuda"] else None
        # initialize asm dict
        self.asm = asm
        # binaries are lazily initialized
        # because it involves doing runtime things
        # (e.g., checking amount of shared memory on current device)
        self.metadata = metadata
        self.cu_module = None
        self.cu_function = None

    def _init_handles(self):
        if self.cu_module is not None:
            return

        if self.device_type in ["cuda"]:
            device = get_current_device()
            bin_path = {
                driver.HIP: "hsaco_path",
                driver.CUDA: "cubin"
            }[driver.backend]
            max_shared = driver.utils.get_device_properties(
                device)["max_shared_mem"]
            fn_load_binary = driver.utils.load_binary
        else:
            assert self.device_backend
            device = self.device_backend.get_current_device()
            bin_path = self.device_backend.get_kernel_bin()
            max_shared = self.device_backend.get_device_properties(
                device)["max_shared_mem"]
            fn_load_binary = self.device_backend.get_load_binary_fn()

        if self.shared > max_shared:
            raise OutOfResources(self.shared, max_shared, "shared memory")

        mod, func, n_regs, n_spills = fn_load_binary(self.metadata["name"],
                                                     self.asm[bin_path],
                                                     self.shared, device)
        # print(f'[FGKCompiledKernel] loading {id(func)}')

        self.n_spills = n_spills
        self.n_regs = n_regs
        self.cu_module = mod
        self.cu_function = func

    def __getattribute__(self, name):
        if name == 'c_wrapper':
            self._init_handles()
        return super().__getattribute__(name)

    # capture args and expand args with cutensormap*
    def assemble_tensormap_to_arg(self, args):
        args_with_tma = list(args)
        if hasattr(self, 'tensormaps_info'):
            # tuple for hashable
            args_ptr = tuple([
                arg.data_ptr() if hasattr(arg, 'data_ptr') else arg
                for arg in args
            ])
            for i, e in enumerate(self.tensormaps_info):
                args_with_tma.append(
                    CompiledKernel.tensormap_manager[(e, args_ptr)])
        return args_with_tma

    def __getitem__(self, grid):
        self._init_handles()

        def runner(*args, stream=None):
            args_expand = self.assemble_tensormap_to_arg(args)
            if stream is None:
                if self.device_type in ["cuda"]:
                    stream = get_cuda_stream()
                else:
                    stream = get_backend(self.device_type).get_stream(None)
            self.c_wrapper(grid[0], grid[1], grid[2], self.num_warps,
                           self.num_ctas, self.clusterDims[0],
                           self.clusterDims[1], self.clusterDims[2],
                           self.shared, stream, self.cu_function,
                           CompiledKernel.launch_enter_hook,
                           CompiledKernel.launch_exit_hook, self, *args_expand)

        return runner
