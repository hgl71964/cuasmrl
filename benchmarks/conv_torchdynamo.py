import os
import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch

import triton
import triton.language as tl

import random
import numpy as np

from cuasmrl.jit import jit
from cuasmrl.autotuner import autotune as fgk_autotune
from cuasmrl.utils.gpu_utils import get_gpu_name, get_gpu_cc

from cuasmrl.autotuner import triton_autotune_with_cache
from cuasmrl.bench import do_bench

# yapf: disable
@dataclass
class Config:
    # Kernel
    default_out_path: str = "data"
    seed: int = 1337
    n_tests: int = 2
    load: Optional[str] = None
    bench: int = 0
    tt: bool = False

    # Workload
    wl: int = 0
    b: int = 32

    # RL
    train: int = 1
    log: int = 1
    verbose: int = 0
    ## Env
    env_id: str = 'cuasmenv-v0'
    num_env: int = 1
    num_iterations: int = int(1e3)
    minibatch_size: int = 8
    horizon: int = 32
    num_steps: int = 64
    normalize_reward: int = 0
    ckpt_freq: int = 100
    ## Agent
    agent: str = "ppo"
    weights_path: Optional[str] = None
    agent_id: Optional[str] = None
    anneal_lr: int = 1
    gae: int = 1
    norm_adv: int = 1
    clip_vloss: int = 1
    update_epochs: int = 4
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    gpu: int = 0


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="???")

    # Add arguments to the parser
    parser.add_argument("--default_out_path", type=str, default="data")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n_tests", type=int, default=2)
    parser.add_argument("--load", type=str)
    parser.add_argument("--bench", type=int, default=0)
    parser.add_argument('--tt', default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--wl", type=int, default=-1)
    parser.add_argument("-b", type=int, default=32)

    parser.add_argument("-t", "--train", type=int, dest="train", default=1)
    parser.add_argument("-l", "--log", type=int, dest="log", default=1)
    parser.add_argument("--verbose", type=int, default=0)

    parser.add_argument("--env_id", type=str, default='cuasmenv-v0')
    parser.add_argument("--num_iterations", type=int, default=int(1e3))
    parser.add_argument("--minibatch_size", type=int, default=8)
    parser.add_argument("--horizon", type=int, dest="horizon", default=32)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--normalize_reward", type=int, default=0)
    parser.add_argument("--ckpt_freq", type=int, default=100)

    parser.add_argument("--agent", type=str, default="ppo")
    parser.add_argument("--weights_path", type=str)
    parser.add_argument("--agent_id", type=str)
    parser.add_argument("--anneal_lr", type=int, default=1)
    parser.add_argument("--gae", type=int, default=1)
    parser.add_argument("--norm_adv", type=int, default=1)
    parser.add_argument("--clip_vloss", type=int, default=1)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    config = Config(**vars(args))
    return config


GPU = get_gpu_name()

# unpack the given idx given the order of axis of the desired 3-dim tensor
# You could view it as the reverse of flatten the idx of 3 axis in a tensor to 1-dim idx.
# order is the order of axes in tensor, innermost dimension outward
# shape is the 3D tensor's shape
def _unpack(idx, order, shape):
    if torch.is_tensor(idx):
        _12 = torch.div(idx, shape[order[0]], rounding_mode="trunc")
        _0 = idx % shape[order[0]]
        _2 = torch.div(_12, shape[order[1]], rounding_mode="trunc")
        _1 = _12 % shape[order[1]]
    else:
        _12 = idx // shape[order[0]]
        _0 = idx % shape[order[0]]
        _2 = _12 // shape[order[1]]
        _1 = _12 % shape[order[1]]
    return _0, _1, _2


class _conv:

    # for the contigous order of w ptr, what"s the corresponding
    # ptr changes for x in a sliding window
    @staticmethod
    def _delta_x_ptr_hwc(
        IN_C, KERNEL_H, KERNEL_W, dilation_h, dilation_w, stride_wc, stride_wh, stride_ww, stride_xc, stride_xh, stride_xw, device,
    ):
        # get the order of axes in w, innermost dimension outward
        stride_w_3d = [stride_wc, stride_wh, stride_ww]
        order = sorted(range(len(stride_w_3d)), key=stride_w_3d.__getitem__)
        window_size = IN_C * KERNEL_H * KERNEL_W

        r_window = torch.arange(0, window_size, 1, device=device)
        window_unpack = _unpack(r_window, order, [IN_C, KERNEL_H, KERNEL_W])
        window_unpack_c = window_unpack[order[0]]
        window_unpack_h = window_unpack[order[1]]
        window_unpack_w = window_unpack[order[2]]
        r_dilation_h = dilation_h * window_unpack_h
        r_dilation_w = dilation_w * window_unpack_w
        r_inc = window_unpack_c
        # delta_x = (
        #     r_dilation_h * stride_xh + r_dilation_w * stride_xw + r_inc * stride_xc
        # )
        # return delta_x
        return (
            r_dilation_h,
            r_dilation_w,
            r_inc,
        )

    @staticmethod
    def _delta_x_ptr(
        IN_C, KERNEL_H, KERNEL_W, dilation_h, dilation_w, stride_wc, stride_wh, stride_ww, stride_xc, stride_xh, stride_xw, device,
    ):
        # get the order of axes in w, innermost dimension outward
        stride_w_3d = [stride_wc, stride_wh, stride_ww]
        order = sorted(range(len(stride_w_3d)), key=stride_w_3d.__getitem__)
        window_size = IN_C * KERNEL_H * KERNEL_W

        r_window = torch.arange(0, window_size, 1, device=device)
        window_unpack = _unpack(r_window, order, [IN_C, KERNEL_H, KERNEL_W])
        window_unpack_c = window_unpack[order[0]]
        window_unpack_h = window_unpack[order[1]]
        window_unpack_w = window_unpack[order[2]]
        r_dilation_h = dilation_h * window_unpack_h
        r_dilation_w = dilation_w * window_unpack_w
        r_inc = window_unpack_c
        delta_x = (
            r_dilation_h * stride_xh + r_dilation_w * stride_xw + r_inc * stride_xc
        )
        return delta_x

@triton.jit
def _kernel_delta_x_hwc(
    x, w, y, # stride of tensor
    stride_xn, stride_xc, stride_xh, stride_xw, stride_wn, stride_wc, stride_wh, stride_ww, stride_yn, stride_yc, stride_yh, stride_yw, stride_biasn, # pointer inc for x
    delta_xh_ptr, delta_xw_ptr, delta_xc_ptr, # Tensor dimensions
    BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W, OUT_H, OUT_W, # parameters of conv
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, output_padding_h, output_padding_w, groups, # 
    # Metaparameters
    ACC_TYPE: tl.constexpr,
    CONV1X1_NHWC: tl.constexpr,
    # blocks in different dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # reduction tiling parameter for matmul
    BLOCK_K: tl.constexpr,
    # Super-blocking for better L2 peformance
    GROUP_H: tl.constexpr,
):
    """
    each program instance computes a [BLOCK_BATCH, BLOCK_N, BLOCK_H, BLOCK_W] block of y
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of y it should compute.
    pid_nhw = tl.program_id(0)
    pid_k = tl.program_id(1)

    # offset for output y
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w

    # offset for the initial ptr for x
    off_x_n = off_y_n
    off_x_h = off_y_h * stride_h - padding_h
    off_x_w = off_y_w * stride_w - padding_w
    off_x_nhw = off_x_n * stride_xn + off_x_h * stride_xh + off_x_w * stride_xw
    off_x_crs = tl.arange(0, BLOCK_K)

    CRS = IN_C * KERNEL_H * KERNEL_W
    # load inc ptr of x, upade x_ptrs
    if not CONV1X1_NHWC:
        delta_xh_ptrs = delta_xh_ptr + off_x_crs
        delta_xw_ptrs = delta_xw_ptr + off_x_crs
        delta_xc_ptrs = delta_xc_ptr + off_x_crs
        delta_xh = tl.load(delta_xh_ptrs, mask=off_x_crs < CRS, other=0)
        delta_xw = tl.load(delta_xw_ptrs, mask=off_x_crs < CRS, other=0)
        delta_xc = tl.load(delta_xc_ptrs, mask=off_x_crs < CRS, other=0)
        off_x_crs_unpacked = (
            delta_xh * stride_xh + delta_xw * stride_xw + delta_xc * stride_xc
        )
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
    else:
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs[None, :]
        delta_xh = 0
        delta_xw = 0

    mask_x = (
        (off_x_n < BATCH)[:, None]
        & (off_x_crs < CRS)[None, :]
        & (off_x_h[:, None] + delta_xh[None, :] >= 0)
        & (off_x_h[:, None] + delta_xh[None, :] < IN_H)
        & (off_x_w[:, None] + delta_xw[None, :] >= 0)
        & (off_x_w[:, None] + delta_xw[None, :] < IN_W)
    )

    # offset for the inital ptr for w
    off_w_crs = tl.arange(0, BLOCK_K)
    off_w_k = off_y_k
    w_ptrs = w + off_w_crs[:, None] + off_w_k[None, :] * stride_wn
    mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]

    # ------ load x ------
    matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
    # ------ load w ------
    matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

    # -----------------------------------------------------------
    # allocate accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for crs in range(0, CRS, BLOCK_K):

        # ------ matrix multiplication ------
        acc += tl.dot(matrix_x, matrix_w)
        # ------ update ptrs ------
        w_ptrs += BLOCK_K
        # load inc ptr of x, upade x_ptrs
        off_x_crs = crs + BLOCK_K + tl.arange(0, BLOCK_K)
        if not CONV1X1_NHWC:
            delta_xh_ptrs += BLOCK_K
            delta_xw_ptrs += BLOCK_K
            delta_xc_ptrs += BLOCK_K
            delta_xh = tl.load(delta_xh_ptrs, mask=off_x_crs < CRS, other=0)
            delta_xw = tl.load(delta_xw_ptrs, mask=off_x_crs < CRS, other=0)
            delta_xc = tl.load(delta_xc_ptrs, mask=off_x_crs < CRS, other=0)
            off_x_crs_unpacked = (
                delta_xh * stride_xh + delta_xw * stride_xw + delta_xc * stride_xc
            )
            x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
        else:
            x_ptrs += BLOCK_K

        mask_x = (
            (off_x_n < BATCH)[:, None]
            & (off_x_crs < CRS)[None, :]
            & (off_x_h[:, None] + delta_xh[None, :] >= 0)
            & (off_x_h[:, None] + delta_xh[None, :] < IN_H)
            & (off_x_w[:, None] + delta_xw[None, :] >= 0)
            & (off_x_w[:, None] + delta_xw[None, :] < IN_W)
        )
        mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]
        # ------ prefetch ------
        # ------ load x ------
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        # ------ load w ------
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

    acc = acc.to(y.dtype.element_ty)

    # rematerialize -- this saves some registers
    # offset for output y
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    # consider output padding
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w

    # y ptrs in the block of [BLOCK_M, BLOCK_N]
    y_ptrs = (
        y
        + off_y_n[:, None] * stride_yn
        + off_y_h[:, None] * stride_yh
        + off_y_w[:, None] * stride_yw
        + off_y_k[None, :] * stride_yc
    )

    # out-of-bounds check
    mask_y = (
        (off_y_n < BATCH)[:, None]
        & (off_y_h < OUT_H + output_padding_h)[:, None]
        & (off_y_w < OUT_W + output_padding_w)[:, None]
        & (off_y_k < KERNEL_N)[None, :]
    )

    tl.store(y_ptrs, acc, mask=mask_y)

@triton.jit
def _kernel_delta_x(
    x, w, y,
    # stride of tensor
    stride_xn, stride_xc, stride_xh, stride_xw, stride_wn, stride_wc, stride_wh, stride_ww, stride_yn, stride_yc, stride_yh, stride_yw, stride_biasn,
    # pointer inc for x
    delta_x_ptr,
    # Tensor dimensions
    BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W, OUT_H, OUT_W,
    # parameters of conv
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, output_padding_h, output_padding_w, groups,
    # Metaparameters
    ACC_TYPE: tl.constexpr,
    CONV1X1_NHWC: tl.constexpr,
    # blocks in different dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # reduction tiling parameter for matmul
    BLOCK_K: tl.constexpr,
    # Super-blocking for better L2 peformance
    GROUP_H: tl.constexpr,
):
    """
    each program instance computes a [BLOCK_BATCH, BLOCK_N, BLOCK_H, BLOCK_W] block of y
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of y it should compute.
    pid_nhw = tl.program_id(0)
    pid_k = tl.program_id(1)

    # offset for output y
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w

    # offset for the initial ptr for x
    off_x_n = off_y_n
    off_x_h = off_y_h * stride_h - padding_h
    off_x_w = off_y_w * stride_w - padding_w
    off_x_nhw = off_x_n * stride_xn + off_x_h * stride_xh + off_x_w * stride_xw
    off_x_crs = tl.arange(0, BLOCK_K)

    CRS = IN_C * KERNEL_H * KERNEL_W
    # load inc ptr of x, upade x_ptrs
    if not CONV1X1_NHWC:
        delta_x_ptrs = delta_x_ptr + off_x_crs
        off_x_crs_unpacked = tl.load(delta_x_ptrs, mask=off_x_crs < CRS)
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
    else:
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs[None, :]

    mask_x = (
        (off_x_n < BATCH)
        & (off_x_h >= 0)
        & (off_x_h < IN_H)
        & (off_x_w >= 0)
        & (off_x_w < IN_W)
    )[:, None] & (off_x_crs < CRS)[None, :]

    # offset for the inital ptr for w
    off_w_crs = tl.arange(0, BLOCK_K)
    off_w_k = off_y_k
    w_ptrs = w + off_w_crs[:, None] + off_w_k[None, :] * stride_wn
    mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]

    # ------ load x ------
    matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
    # ------ load w ------
    matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

    # -----------------------------------------------------------
    # allocate accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for crs in range(0, CRS, BLOCK_K):

        # ------ matrix multiplication ------
        acc += tl.dot(matrix_x, matrix_w)
        # ------ update ptrs ------
        w_ptrs += BLOCK_K
        # load inc ptr of x, upade x_ptrs
        if not CONV1X1_NHWC:
            delta_x_ptrs += BLOCK_K
            off_x_crs = crs + BLOCK_K + tl.arange(0, BLOCK_K)
            off_x_crs_unpacked = tl.load(delta_x_ptrs, mask=off_x_crs < CRS, other=0)
            x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
        else:
            off_x_crs = crs + BLOCK_K + tl.arange(0, BLOCK_K)
            x_ptrs += BLOCK_K

        mask_x = (
            (off_x_n < BATCH)
            & (off_x_h >= 0)
            & (off_x_h < IN_H)
            & (off_x_w >= 0)
            & (off_x_w < IN_W)
        )[:, None] & (off_x_crs < CRS)[None, :]
        mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]
        # ------ prefetch ------
        # ------ load x ------
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        # ------ load w ------
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

    acc = acc.to(y.dtype.element_ty)

    # rematerialize -- this saves some registers
    # offset for output y
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    # consider output padding
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w

    # y ptrs in the block of [BLOCK_M, BLOCK_N]
    y_ptrs = (
        y
        + off_y_n[:, None] * stride_yn
        + off_y_h[:, None] * stride_yh
        + off_y_w[:, None] * stride_yw
        + off_y_k[None, :] * stride_yc
    )

    # out-of-bounds check
    mask_y = (
        (off_y_n < BATCH)[:, None]
        & (off_y_h < OUT_H + output_padding_h)[:, None]
        & (off_y_w < OUT_W + output_padding_w)[:, None]
        & (off_y_k < KERNEL_N)[None, :]
    )

    tl.store(y_ptrs, acc, mask=mask_y)

def forward(
        k1,
        k2,
        x,
        w,
        bias,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        transposed=False,
        output_padding=(0, 0),
        groups=1,
        load_dir=None,
):
    if groups != 1:
        raise RuntimeError("groups must be 1")
    if transposed:
        raise RuntimeError("transposed must be False")

    # Q: should we check x, w, bias dtypes?
    device = x.device
    # input shapes
    shape_x = x.shape
    shape_w = w.shape
    shape_bias = bias.shape if bias is not None else None

    # indicies for the layeout
    xn, xc, xh, xw = 0, 1, 2, 3
    yn, yc, yh, yw = 0, 1, 2, 3
    wn, wc, wh, ww = 0, 1, 2, 3

    # out_channel, in_channel, kernel_height, kernel_width
    kernel_size = [shape_w[wh], shape_w[ww]]
    input_size = [shape_x[xh], shape_x[xw]]
    assert (not shape_bias or shape_bias[0] == shape_w[wn]
            ), f"bias shape did not match{shape_bias} != {shape_w[wn]}"
    in_channel = shape_w[wc] * groups

    assert shape_x[xc] % groups == 0, "in_channels must be divisible by groups"
    assert shape_w[wn] % groups == 0, "out_channels must be divisible by groups"
    assert (shape_x[xc] == in_channel
            ), f"in_channel did not match {shape_x[xc]} != {in_channel}"

    assert (len(stride) == len(padding) == len(dilation) == len(output_padding)
            == len(kernel_size) == len(input_size))

    # output shape
    shape_y = [0] * 4
    shape_y[yn] = shape_x[xn]
    shape_y[yc] = shape_w[wn]
    shape_y[yh] = (input_size[0] + 2 * padding[0] - dilation[0] *
                   (kernel_size[0] - 1) - 1 +
                   stride[0]) // stride[0] + 2 * output_padding[0]
    shape_y[yw] = (input_size[1] + 2 * padding[1] - dilation[1] *
                   (kernel_size[1] - 1) - 1 +
                   stride[1]) // stride[1] + 2 * output_padding[1]

    BATCH = shape_x[xn]
    IN_C = shape_x[xc]
    IN_H = shape_x[xh]
    IN_W = shape_x[xw]
    KERNEL_N = shape_w[wn]
    KERNEL_H = shape_w[wh]
    KERNEL_W = shape_w[ww]
    OUT_H = shape_y[yh]
    OUT_W = shape_y[yw]

    # allocate output
    y = torch.empty(shape_y, device=device, dtype=x.dtype)

    # get strides for tensors
    stride_x = x.stride()
    stride_w = w.stride()
    stride_bias = bias.stride() if shape_bias else None
    stride_biasn = stride_bias[0] if stride_bias else None

    # output layout should be the same as x
    if stride_x[xc] < stride_x[xh] and stride_x[xc] < stride_x[xw]:
        y = y.to(memory_format=torch.channels_last)
    stride_y = y.stride()

    # allocate tmp
    # WINDOW_SIZE = KERNEL_H * KERNEL_W * IN_C
    # tmp_x = torch.empty((BATCH * OUT_H * OUT_W, WINDOW_SIZE), device=device, dtype=x.dtype)
    # tmp_w = torch.empty((WINDOW_SIZE, KERNEL_N), device=device, dtype=w.dtype)
    # accumulator types
    ACC_TYPE = (tl.float32 if x.dtype in [
        torch.float16, torch.bfloat16, torch.float32
    ] else tl.int32)
    # if stride_x[xc] == 1 and stride_x > 1 and stride_y > 1:
    CONV1X1_NHWC = False
    if stride_x[xc] == 1 and KERNEL_H == 1 and KERNEL_W == 1:
        CONV1X1_NHWC = True
    #  do we need delta x ptr for h, w, c dimension each or not
    DELTA_X_PTR_HWC = (False if ((padding[0] == 0 and padding[1] == 0) or
                                 (KERNEL_H == 1 and KERNEL_W == 1)) else True)
    if not CONV1X1_NHWC:
        if DELTA_X_PTR_HWC:
            delta_xh, delta_xw, delta_xc = _conv._delta_x_ptr_hwc(
                IN_C,
                KERNEL_H,
                KERNEL_W,
                dilation[0],
                dilation[1],
                stride_w[wc],
                stride_w[wh],
                stride_w[ww],
                stride_x[xc],
                stride_x[xh],
                stride_x[xw],
                device,
            )
        else:
            delta_x = _conv._delta_x_ptr(
                IN_C,
                KERNEL_H,
                KERNEL_W,
                dilation[0],
                dilation[1],
                stride_w[wc],
                stride_w[wh],
                stride_w[ww],
                stride_x[xc],
                stride_x[xh],
                stride_x[xw],
                device,
            )
    else:
        delta_x = None
        delta_xh, delta_xw, delta_xc = None, None, None

    # launch kernel, 2-dim, batch*h*w, kernel
    def grid(META):
        return (
            triton.cdiv(BATCH * OUT_H * OUT_W, META["BLOCK_M"]),
            triton.cdiv(KERNEL_N, META["BLOCK_N"]),
        )

    # conv1x1 or padding==0
    if CONV1X1_NHWC or not DELTA_X_PTR_HWC:
        print('k1::')
        k1[grid](
            x, w, y, #
            # stride nchw for x,w,y tensor
            stride_x[xn], stride_x[xc], stride_x[xh], stride_x[xw], stride_w[wn], stride_w[wc], stride_w[wh], stride_w[ww], stride_y[yn], stride_y[yc], stride_y[yh], stride_y[yw], stride_biasn,  #
            # pointer inc for x
            delta_x,
            # Tensor dimensions
            BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W, OUT_H, OUT_W,  #
            # conv parameters
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1], output_padding[0], output_padding[1], groups,  #
            # Metaparameters
            # ACC_TYPE=ACC_TYPE,
            # CONV1X1_NHWC=CONV1X1_NHWC,
            # BLOCK_M=128,
            # BLOCK_N=32,
            # BLOCK_K=32,
            # GROUP_H=1,
            # BLOCK_M = 256,
            # BLOCK_N = 32,
            # BLOCK_K = 64,
            # num_stages=4,
            # num_warps=4,
            load_dir=load_dir,
        )
    # need to know ptr update for each dimension to check if
    # the sliding window is out of bounds
    else:
        print('k2::')
        k2[grid](
            x, w, y,  # 
            # stride nchw for x,w,y tensor
            stride_x[xn], stride_x[xc], stride_x[xh], stride_x[xw], stride_w[wn], stride_w[wc], stride_w[wh], stride_w[ww], stride_y[yn], stride_y[yc], stride_y[yh], stride_y[yw], stride_biasn,  # 
            # pointer inc for x
            delta_xh, delta_xw, delta_xc, #
            # Tensor dimensions
            BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W, OUT_H, OUT_W,  # 
            # conv parameters
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1], output_padding[0], output_padding[1], groups, # 
            # Metaparameters
            # ACC_TYPE=ACC_TYPE,
            # CONV1X1_NHWC=CONV1X1_NHWC,
            # BLOCK_M=128,
            # BLOCK_N=32,
            # BLOCK_K=32,
            # GROUP_H=1,

            # BLOCK_M = 256,
            # BLOCK_N = 32,
            # BLOCK_K = 64,
            # num_stages=4,
            # num_warps=4,
            load_dir=load_dir,
        )

    if bias is not None:
        if len(bias.shape) == 1:
            bias = bias.reshape([1, bias.shape[0], 1, 1])
        y += bias
    return y


def forward_tt(
        k1,
        k2,
        x,
        w,
        bias,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        transposed=False,
        output_padding=(0, 0),
        groups=1,
):
    if groups != 1:
        raise RuntimeError("groups must be 1")
    if transposed:
        raise RuntimeError("transposed must be False")

    # Q: should we check x, w, bias dtypes?
    device = x.device
    # input shapes
    shape_x = x.shape
    shape_w = w.shape
    shape_bias = bias.shape if bias is not None else None

    # indicies for the layeout
    xn, xc, xh, xw = 0, 1, 2, 3
    yn, yc, yh, yw = 0, 1, 2, 3
    wn, wc, wh, ww = 0, 1, 2, 3

    # out_channel, in_channel, kernel_height, kernel_width
    kernel_size = [shape_w[wh], shape_w[ww]]
    input_size = [shape_x[xh], shape_x[xw]]
    assert (not shape_bias or shape_bias[0] == shape_w[wn]
            ), f"bias shape did not match{shape_bias} != {shape_w[wn]}"
    in_channel = shape_w[wc] * groups

    assert shape_x[xc] % groups == 0, "in_channels must be divisible by groups"
    assert shape_w[wn] % groups == 0, "out_channels must be divisible by groups"
    assert (shape_x[xc] == in_channel
            ), f"in_channel did not match {shape_x[xc]} != {in_channel}"

    assert (len(stride) == len(padding) == len(dilation) == len(output_padding)
            == len(kernel_size) == len(input_size))

    # output shape
    shape_y = [0] * 4
    shape_y[yn] = shape_x[xn]
    shape_y[yc] = shape_w[wn]
    shape_y[yh] = (input_size[0] + 2 * padding[0] - dilation[0] *
                   (kernel_size[0] - 1) - 1 +
                   stride[0]) // stride[0] + 2 * output_padding[0]
    shape_y[yw] = (input_size[1] + 2 * padding[1] - dilation[1] *
                   (kernel_size[1] - 1) - 1 +
                   stride[1]) // stride[1] + 2 * output_padding[1]

    BATCH = shape_x[xn]
    IN_C = shape_x[xc]
    IN_H = shape_x[xh]
    IN_W = shape_x[xw]
    KERNEL_N = shape_w[wn]
    KERNEL_H = shape_w[wh]
    KERNEL_W = shape_w[ww]
    OUT_H = shape_y[yh]
    OUT_W = shape_y[yw]

    # allocate output
    y = torch.empty(shape_y, device=device, dtype=x.dtype)

    # get strides for tensors
    stride_x = x.stride()
    stride_w = w.stride()
    stride_bias = bias.stride() if shape_bias else None
    stride_biasn = stride_bias[0] if stride_bias else None

    # output layout should be the same as x
    if stride_x[xc] < stride_x[xh] and stride_x[xc] < stride_x[xw]:
        y = y.to(memory_format=torch.channels_last)
    stride_y = y.stride()

    # allocate tmp
    # WINDOW_SIZE = KERNEL_H * KERNEL_W * IN_C
    # tmp_x = torch.empty((BATCH * OUT_H * OUT_W, WINDOW_SIZE), device=device, dtype=x.dtype)
    # tmp_w = torch.empty((WINDOW_SIZE, KERNEL_N), device=device, dtype=w.dtype)
    # accumulator types
    ACC_TYPE = (tl.float32 if x.dtype in [
        torch.float16, torch.bfloat16, torch.float32
    ] else tl.int32)
    # if stride_x[xc] == 1 and stride_x > 1 and stride_y > 1:
    CONV1X1_NHWC = False
    if stride_x[xc] == 1 and KERNEL_H == 1 and KERNEL_W == 1:
        CONV1X1_NHWC = True
    #  do we need delta x ptr for h, w, c dimension each or not
    DELTA_X_PTR_HWC = (False if ((padding[0] == 0 and padding[1] == 0) or
                                 (KERNEL_H == 1 and KERNEL_W == 1)) else True)
    if not CONV1X1_NHWC:
        if DELTA_X_PTR_HWC:
            delta_xh, delta_xw, delta_xc = _conv._delta_x_ptr_hwc(
                IN_C,
                KERNEL_H,
                KERNEL_W,
                dilation[0],
                dilation[1],
                stride_w[wc],
                stride_w[wh],
                stride_w[ww],
                stride_x[xc],
                stride_x[xh],
                stride_x[xw],
                device,
            )
        else:
            delta_x = _conv._delta_x_ptr(
                IN_C,
                KERNEL_H,
                KERNEL_W,
                dilation[0],
                dilation[1],
                stride_w[wc],
                stride_w[wh],
                stride_w[ww],
                stride_x[xc],
                stride_x[xh],
                stride_x[xw],
                device,
            )
    else:
        delta_x = None
        delta_xh, delta_xw, delta_xc = None, None, None

    # launch kernel, 2-dim, batch*h*w, kernel
    def grid(META):
        return (
            triton.cdiv(BATCH * OUT_H * OUT_W, META["BLOCK_M"]),
            triton.cdiv(KERNEL_N, META["BLOCK_N"]),
        )

    # conv1x1 or padding==0
    if CONV1X1_NHWC or not DELTA_X_PTR_HWC:
        print('k1::')
        k1[grid](
            x, w, y, #
            # stride nchw for x,w,y tensor
            stride_x[xn], stride_x[xc], stride_x[xh], stride_x[xw], stride_w[wn], stride_w[wc], stride_w[wh], stride_w[ww], stride_y[yn], stride_y[yc], stride_y[yh], stride_y[yw], stride_biasn,  #
            # pointer inc for x
            delta_x,
            # Tensor dimensions
            BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W, OUT_H, OUT_W,  #
            # conv parameters
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1], output_padding[0], output_padding[1], groups,  #
            # Metaparameters
            # ACC_TYPE=ACC_TYPE,
            # CONV1X1_NHWC=CONV1X1_NHWC,
            # BLOCK_M=128,
            # BLOCK_N=32,
            # BLOCK_K=32,
            # GROUP_H=1,
            # BLOCK_M = 256,
            # BLOCK_N = 32,
            # BLOCK_K = 64,
            # num_stages=4,
            # num_warps=4,
        )
    # need to know ptr update for each dimension to check if
    # the sliding window is out of bounds
    else:
        print('k2::')
        k2[grid](
            x, w, y,  # 
            # stride nchw for x,w,y tensor
            stride_x[xn], stride_x[xc], stride_x[xh], stride_x[xw], stride_w[wn], stride_w[wc], stride_w[wh], stride_w[ww], stride_y[yn], stride_y[yc], stride_y[yh], stride_y[yw], stride_biasn,  # 
            # pointer inc for x
            delta_xh, delta_xw, delta_xc, #
            # Tensor dimensions
            BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W, OUT_H, OUT_W,  # 
            # conv parameters
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1], output_padding[0], output_padding[1], groups, # 
            # Metaparameters
            # ACC_TYPE=ACC_TYPE,
            # CONV1X1_NHWC=CONV1X1_NHWC,
            # BLOCK_M=128,
            # BLOCK_N=32,
            # BLOCK_K=32,
            # GROUP_H=1,

            # BLOCK_M = 256,
            # BLOCK_N = 32,
            # BLOCK_K = 64,
            # num_stages=4,
            # num_warps=4,
        )

    if bias is not None:
        if len(bias.shape) == 1:
            bias = bias.reshape([1, bias.shape[0], 1, 1])
        y += bias
    return y




def main():
    drl_config = parse_args()
    random.seed(drl_config.seed)
    np.random.seed(drl_config.seed)
    torch.manual_seed(drl_config.seed)

    resnet50_layers = (
        # IN_H, IN_W, IN_C, KERNEL_H, KERNEL_W, KERNEL_N, stride, padding
        (224, 224, 3, 7, 7, 64, (2, 2), (0, 0)),
        # conv2_x
        (56, 56, 64, 1, 1, 64, (1, 1), (0, 0)),
        (56, 56, 64, 3, 3, 64, (1, 1), (0, 0)),
        (56, 56, 64, 1, 1, 256, (1, 1), (0, 0)),
        # conv3_x
        (56, 56, 256, 1, 1, 128, (2, 2), (0, 0)),
        (28, 28, 128, 3, 3, 128, (1, 1), (0, 0)),
        (28, 28, 128, 1, 1, 512, (1, 1), (0, 0)),
        # conv4_x
        (28, 28, 512, 1, 1, 256, (2, 2), (0, 0)),
        (14, 14, 256, 3, 3, 256, (1, 1), (0, 0)),
        (14, 14, 256, 1, 1, 1024, (1, 1), (0, 0)),
        # conv5_x
        (14, 14, 1024, 1, 1, 512, (2, 2), (0, 0)),
        (7, 7, 512, 3, 3, 512, (1, 1), (0, 0)),
        (7, 7, 512, 1, 1, 2048, (1, 1), (0, 0)),
    )

    # workload
    BATCH = drl_config.b
    wl = drl_config.wl
    assert wl < len(resnet50_layers), f"wl {wl} > {len(resnet50_layers)}"
    dtype = torch.float32
    IN_H, IN_W, IN_C, KERNEL_H, KERNEL_W, KERNEL_N, stride, padding = resnet50_layers[
        wl]
    layout = "nhwc"
    dilation = (1, 1)
    groups = 1

    # allocate inputs, nchw
    x = torch.randn((BATCH, IN_C, IN_H, IN_W), dtype=dtype, device="cuda")
    w = torch.randn((KERNEL_N, IN_C // groups, KERNEL_H, KERNEL_W),
                    dtype=dtype,
                    device="cuda")
    bias = torch.randn((KERNEL_N), dtype=dtype, device="cuda")
    if layout == "nhwc":
        x = x.to(memory_format=torch.channels_last)
        w = w.to(memory_format=torch.channels_last)
    OUT_H = (IN_H + 2 * padding[0] - dilation[0] *
             (KERNEL_H - 1) - 1 + stride[0]) // stride[0]
    OUT_W = (IN_W + 2 * padding[1] - dilation[1] *
             (KERNEL_W - 1) - 1 + stride[1]) // stride[1]

    flops = 2.0 * BATCH * OUT_H * OUT_W * IN_C * KERNEL_H * KERNEL_W * KERNEL_N

    drl_config.total_flops = flops
    drl_config.save_dir = f'{GPU}/conv/{wl}'
    if drl_config.load is None:
        load_dir = None
    elif drl_config.load == "auto":
        load_dir = f'data/{GPU}/conv/{wl}'
    else:
        load_dir = drl_config.load


    @fgk_autotune(
        configs=[
            triton.Config({'ACC_TYPE': tl.float32, 'CONV1X1_NHWC': True, 'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
    ],
        key=['BLOCK_M'],
        ret_ptr=2,
        drl_config=drl_config,
    )
    @jit
    def _cuasmrl_k2(
        x, w, y, # stride of tensor
        stride_xn, stride_xc, stride_xh, stride_xw, stride_wn, stride_wc, stride_wh, stride_ww, stride_yn, stride_yc, stride_yh, stride_yw, stride_biasn, # pointer inc for x
        delta_xh_ptr, delta_xw_ptr, delta_xc_ptr, # Tensor dimensions
        BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W, OUT_H, OUT_W, # parameters of conv
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, output_padding_h, output_padding_w, groups, # 
        # Metaparameters
        ACC_TYPE: tl.constexpr,
        CONV1X1_NHWC: tl.constexpr,
        # blocks in different dimension
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        # reduction tiling parameter for matmul
        BLOCK_K: tl.constexpr,
        # Super-blocking for better L2 peformance
        # GROUP_H: tl.constexpr,
    ):
        """
        each program instance computes a [BLOCK_BATCH, BLOCK_N, BLOCK_H, BLOCK_W] block of y
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of y it should compute.
        pid_nhw = tl.program_id(0)
        pid_k = tl.program_id(1)

        # offset for output y
        off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
        off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
        off_y_n = off_y_nhw // (OUT_H * OUT_W)
        off_y_hw = off_y_nhw % (OUT_H * OUT_W)
        off_y_h = off_y_hw // OUT_W + output_padding_h
        off_y_w = off_y_hw % OUT_W + output_padding_w

        # offset for the initial ptr for x
        off_x_n = off_y_n
        off_x_h = off_y_h * stride_h - padding_h
        off_x_w = off_y_w * stride_w - padding_w
        off_x_nhw = off_x_n * stride_xn + off_x_h * stride_xh + off_x_w * stride_xw
        off_x_crs = tl.arange(0, BLOCK_K)

        CRS = IN_C * KERNEL_H * KERNEL_W
        # load inc ptr of x, upade x_ptrs
        if not CONV1X1_NHWC:
            delta_xh_ptrs = delta_xh_ptr + off_x_crs
            delta_xw_ptrs = delta_xw_ptr + off_x_crs
            delta_xc_ptrs = delta_xc_ptr + off_x_crs
            delta_xh = tl.load(delta_xh_ptrs, mask=off_x_crs < CRS, other=0)
            delta_xw = tl.load(delta_xw_ptrs, mask=off_x_crs < CRS, other=0)
            delta_xc = tl.load(delta_xc_ptrs, mask=off_x_crs < CRS, other=0)
            off_x_crs_unpacked = (
                delta_xh * stride_xh + delta_xw * stride_xw + delta_xc * stride_xc
            )
            x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
        else:
            x_ptrs = x + off_x_nhw[:, None] + off_x_crs[None, :]
            delta_xh = 0
            delta_xw = 0

        mask_x = (
            (off_x_n < BATCH)[:, None]
            & (off_x_crs < CRS)[None, :]
            & (off_x_h[:, None] + delta_xh[None, :] >= 0)
            & (off_x_h[:, None] + delta_xh[None, :] < IN_H)
            & (off_x_w[:, None] + delta_xw[None, :] >= 0)
            & (off_x_w[:, None] + delta_xw[None, :] < IN_W)
        )

        # offset for the inital ptr for w
        off_w_crs = tl.arange(0, BLOCK_K)
        off_w_k = off_y_k
        w_ptrs = w + off_w_crs[:, None] + off_w_k[None, :] * stride_wn
        mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]

        # ------ load x ------
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        # ------ load w ------
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

        # -----------------------------------------------------------
        # allocate accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
        for crs in range(0, CRS, BLOCK_K):

            # ------ matrix multiplication ------
            acc += tl.dot(matrix_x, matrix_w)
            # ------ update ptrs ------
            w_ptrs += BLOCK_K
            # load inc ptr of x, upade x_ptrs
            off_x_crs = crs + BLOCK_K + tl.arange(0, BLOCK_K)
            if not CONV1X1_NHWC:
                delta_xh_ptrs += BLOCK_K
                delta_xw_ptrs += BLOCK_K
                delta_xc_ptrs += BLOCK_K
                delta_xh = tl.load(delta_xh_ptrs, mask=off_x_crs < CRS, other=0)
                delta_xw = tl.load(delta_xw_ptrs, mask=off_x_crs < CRS, other=0)
                delta_xc = tl.load(delta_xc_ptrs, mask=off_x_crs < CRS, other=0)
                off_x_crs_unpacked = (
                    delta_xh * stride_xh + delta_xw * stride_xw + delta_xc * stride_xc
                )
                x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
            else:
                x_ptrs += BLOCK_K

            mask_x = (
                (off_x_n < BATCH)[:, None]
                & (off_x_crs < CRS)[None, :]
                & (off_x_h[:, None] + delta_xh[None, :] >= 0)
                & (off_x_h[:, None] + delta_xh[None, :] < IN_H)
                & (off_x_w[:, None] + delta_xw[None, :] >= 0)
                & (off_x_w[:, None] + delta_xw[None, :] < IN_W)
            )
            mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]
            # ------ prefetch ------
            # ------ load x ------
            matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
            # ------ load w ------
            matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = acc.to(y.dtype.element_ty)

        # rematerialize -- this saves some registers
        # offset for output y
        off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
        off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
        off_y_n = off_y_nhw // (OUT_H * OUT_W)
        off_y_hw = off_y_nhw % (OUT_H * OUT_W)
        # consider output padding
        off_y_h = off_y_hw // OUT_W + output_padding_h
        off_y_w = off_y_hw % OUT_W + output_padding_w

        # y ptrs in the block of [BLOCK_M, BLOCK_N]
        y_ptrs = (
            y
            + off_y_n[:, None] * stride_yn
            + off_y_h[:, None] * stride_yh
            + off_y_w[:, None] * stride_yw
            + off_y_k[None, :] * stride_yc
        )

        # out-of-bounds check
        mask_y = (
            (off_y_n < BATCH)[:, None]
            & (off_y_h < OUT_H + output_padding_h)[:, None]
            & (off_y_w < OUT_W + output_padding_w)[:, None]
            & (off_y_k < KERNEL_N)[None, :]
        )

        tl.store(y_ptrs, acc, mask=mask_y)

    @fgk_autotune(
        configs=[
            triton.Config({'ACC_TYPE': tl.float32, 'CONV1X1_NHWC': True, 'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
    ],
        key=['BLOCK_M'],
        ret_ptr=2,
        drl_config=drl_config,
    )
    @jit
    def _cuasmrl_k1(
        x, w, y,
        # stride of tensor
        stride_xn, stride_xc, stride_xh, stride_xw, stride_wn, stride_wc, stride_wh, stride_ww, stride_yn, stride_yc, stride_yh, stride_yw, stride_biasn,
        # pointer inc for x
        delta_x_ptr,
        # Tensor dimensions
        BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W, OUT_H, OUT_W,
        # parameters of conv
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, output_padding_h, output_padding_w, groups,
        # Metaparameters
        ACC_TYPE: tl.constexpr,
        CONV1X1_NHWC: tl.constexpr,
        # blocks in different dimension
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        # reduction tiling parameter for matmul
        BLOCK_K: tl.constexpr,
        # Super-blocking for better L2 peformance
        # GROUP_H: tl.constexpr,
    ):
        """
        each program instance computes a [BLOCK_BATCH, BLOCK_N, BLOCK_H, BLOCK_W] block of y
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of y it should compute.
        pid_nhw = tl.program_id(0)
        pid_k = tl.program_id(1)

        # offset for output y
        off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
        off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
        off_y_n = off_y_nhw // (OUT_H * OUT_W)
        off_y_hw = off_y_nhw % (OUT_H * OUT_W)
        off_y_h = off_y_hw // OUT_W + output_padding_h
        off_y_w = off_y_hw % OUT_W + output_padding_w

        # offset for the initial ptr for x
        off_x_n = off_y_n
        off_x_h = off_y_h * stride_h - padding_h
        off_x_w = off_y_w * stride_w - padding_w
        off_x_nhw = off_x_n * stride_xn + off_x_h * stride_xh + off_x_w * stride_xw
        off_x_crs = tl.arange(0, BLOCK_K)

        CRS = IN_C * KERNEL_H * KERNEL_W
        # load inc ptr of x, upade x_ptrs
        if not CONV1X1_NHWC:
            delta_x_ptrs = delta_x_ptr + off_x_crs
            off_x_crs_unpacked = tl.load(delta_x_ptrs, mask=off_x_crs < CRS)
            x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
        else:
            x_ptrs = x + off_x_nhw[:, None] + off_x_crs[None, :]

        mask_x = (
            (off_x_n < BATCH)
            & (off_x_h >= 0)
            & (off_x_h < IN_H)
            & (off_x_w >= 0)
            & (off_x_w < IN_W)
        )[:, None] & (off_x_crs < CRS)[None, :]

        # offset for the inital ptr for w
        off_w_crs = tl.arange(0, BLOCK_K)
        off_w_k = off_y_k
        w_ptrs = w + off_w_crs[:, None] + off_w_k[None, :] * stride_wn
        mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]

        # ------ load x ------
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        # ------ load w ------
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

        # -----------------------------------------------------------
        # allocate accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
        for crs in range(0, CRS, BLOCK_K):

            # ------ matrix multiplication ------
            acc += tl.dot(matrix_x, matrix_w)
            # ------ update ptrs ------
            w_ptrs += BLOCK_K
            # load inc ptr of x, upade x_ptrs
            if not CONV1X1_NHWC:
                delta_x_ptrs += BLOCK_K
                off_x_crs = crs + BLOCK_K + tl.arange(0, BLOCK_K)
                off_x_crs_unpacked = tl.load(delta_x_ptrs, mask=off_x_crs < CRS, other=0)
                x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
            else:
                off_x_crs = crs + BLOCK_K + tl.arange(0, BLOCK_K)
                x_ptrs += BLOCK_K

            mask_x = (
                (off_x_n < BATCH)
                & (off_x_h >= 0)
                & (off_x_h < IN_H)
                & (off_x_w >= 0)
                & (off_x_w < IN_W)
            )[:, None] & (off_x_crs < CRS)[None, :]
            mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]
            # ------ prefetch ------
            # ------ load x ------
            matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
            # ------ load w ------
            matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = acc.to(y.dtype.element_ty)

        # rematerialize -- this saves some registers
        # offset for output y
        off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
        off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
        off_y_n = off_y_nhw // (OUT_H * OUT_W)
        off_y_hw = off_y_nhw % (OUT_H * OUT_W)
        # consider output padding
        off_y_h = off_y_hw // OUT_W + output_padding_h
        off_y_w = off_y_hw % OUT_W + output_padding_w

        # y ptrs in the block of [BLOCK_M, BLOCK_N]
        y_ptrs = (
            y
            + off_y_n[:, None] * stride_yn
            + off_y_h[:, None] * stride_yh
            + off_y_w[:, None] * stride_yw
            + off_y_k[None, :] * stride_yc
        )

        # out-of-bounds check
        mask_y = (
            (off_y_n < BATCH)[:, None]
            & (off_y_h < OUT_H + output_padding_h)[:, None]
            & (off_y_w < OUT_W + output_padding_w)[:, None]
            & (off_y_k < KERNEL_N)[None, :]
        )

        tl.store(y_ptrs, acc, mask=mask_y)


    tri_out = forward(
        _cuasmrl_k1,
        _cuasmrl_k2,
        x,
        w,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        transposed=False,
        # output_padding=(0, 0),
        groups=1,
        load_dir=load_dir,
    )


    if drl_config.tt:
        tri_out = forward_tt(
            _kernel_delta_x,
            _kernel_delta_x_hwc,
            x,
            w,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            transposed=False,
            # output_padding=(0, 0),
            groups=1,
        )


if __name__ == "__main__":
    main()
