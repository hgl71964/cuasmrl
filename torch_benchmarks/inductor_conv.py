import torch
from torch import _dynamo as torchdynamo
# from torch import _inductor as inductor

# run to dump kernel: TORCH_COMPILE_DEBUG=1 python conv.py

device = torch.device("cuda:0")

batch_size = 32


def conv(x):
    x = torch.nn.Conv2d(3, 6, kernel_size=3, padding=1).to(device)(x)
    return x


# TODO or equally?
@torchdynamo.optimize("inductor", nopython=True)
def inductor_conv(x):
    x = torch.nn.Conv2d(3, 6, kernel_size=3, padding=1).to(device)(x)
    return x


# https://pytorch.org/docs/stable/generated/torch.compile.html#torch.compile
compiled_model = torch.compile(
    conv,
    backend='inductor',
    mode='max-autotune-no-cudagraphs',
)

x = torch.randn(batch_size, 3, 224, 224).to(device)
y = compiled_model(x)
