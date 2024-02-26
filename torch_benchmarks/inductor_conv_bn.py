import torch
# from torch import _dynamo as torchdynamo
# from torch import _inductor as inductor
# from torch._inductor.triton_heuristics import AutotuneHint, pointwise

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("n", 0, "")

# resnet50 layer shape
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

alexnet_layers = (
    # IN_H, IN_W, IN_C, KERNEL_H, KERNEL_W, KERNEL_N, stride, padding
    (224, 224, 3, 11, 11, 64, (4, 4), (2, 2)), )


def conv_bn_relu_torchinductor(
    x,
    w,
    bias,
    stride,
    padding,
    dilation,
    groups,
    running_mean,
    running_var,
    bn_weight,
    bn_bias,
    device,
):
    # y = torch.conv2d(x, w, None, stride, padding, dilation, groups)

    y = torch.nn.Conv2d(
        x.shape[1],
        w.shape[0],
        kernel_size=w.shape[2:],
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    ).to(device)(x)

    y = torch.batch_norm(
        y,
        weight=bn_weight,
        bias=bn_bias,
        running_mean=running_mean,
        running_var=running_var,
        training=False,
        momentum=1,
        eps=1e-5,
        cudnn_enabled=True,
    )
    y = torch.nn.ReLU().to(device)(y)
    return y


def main(_):
    device = torch.device("cuda:0")

    # workload
    BATCH = 32
    layer = resnet50_layers[FLAGS.n]
    IN_H, IN_W, IN_C, KERNEL_H, KERNEL_W, KERNEL_N, stride, padding = layer
    dilation, groups = (1, 1), 1
    dtype = torch.float32

    OUT_H = (IN_H + 2 * padding[0] - dilation[0] *
             (KERNEL_H - 1) - 1 + stride[0]) // stride[0]
    OUT_W = (IN_W + 2 * padding[1] - dilation[1] *
             (KERNEL_W - 1) - 1 + stride[1]) // stride[1]
    tflops = (lambda ms: 2.0 * BATCH * OUT_H * OUT_W * IN_C * KERNEL_H *
              KERNEL_W * KERNEL_N / ms * 1e-9)

    # allocate inputs, nchw
    x = torch.randn((BATCH, IN_C, IN_H, IN_W), dtype=dtype).to(device)
    w = torch.randn(
        (KERNEL_N, IN_C // groups, KERNEL_H, KERNEL_W),
        dtype=dtype,
    ).to(device)

    bias = torch.randn((KERNEL_N, ), dtype=dtype).to(device)

    args = (x, w, bias, stride, padding, dilation, groups)

    running_mean = torch.randn(
        (KERNEL_N),
        dtype=dtype,
    ).to(device)
    running_var = torch.randn(
        (KERNEL_N),
        dtype=dtype,
    ).to(device)
    bn_weight = torch.randn(
        (KERNEL_N),
        dtype=dtype,
    ).to(device)
    bn_bias = torch.randn(
        (KERNEL_N),
        dtype=dtype,
    ).to(device)
    args += (
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
        device,
    )

    # https://pytorch.org/docs/stable/generated/torch.compile.html#torch.compile
    fn = torch.compile(
        conv_bn_relu_torchinductor,
        backend='inductor',
        mode='max-autotune-no-cudagraphs',
    )

    fn(*args)


if __name__ == "__main__":
    app.run(main)
