import torch
import triton
from torch import _dynamo as torchdynamo

from absl import app
from absl import flags

# see: TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS
# from torch._inductor.config
# also: https://discuss.pytorch.org/t/getting-triton-to-generate-all-kernels/189173

FLAGS = flags.FLAGS
flags.DEFINE_integer("n", 0, "")


# @torchdynamo.optimize("inductor", nopython=True)
def mm_add_relu(a, b, bias):
    y = torch.mm(a, b)
    y += bias
    return torch.relu(y)


def bench(shape):
    dtype = torch.float16
    M, K = shape[0]
    _, N = shape[1]
    torch.manual_seed(0)
    # allocate inputs
    a = torch.randn(shape[0], device="cuda", dtype=dtype)
    b = torch.randn(shape[1], device="cuda", dtype=dtype)

    def tflops(ms):
        return M * K * N / ms * 1e-9

    bias = torch.randn((M, N), dtype=dtype, device="cuda")

    args = (a, b, bias)

    # https://pytorch.org/docs/stable/generated/torch.compile.html#torch.compile
    fn = torch.compile(
        mm_add_relu,
        backend='inductor',
        mode='max-autotune-no-cudagraphs',
    )

    fn(*args)


def main(_):
    # torchinductor.config.debug = True
    torch.manual_seed(0)

    # The flag below controls whether to allow TF32 on matmul.
    torch.backends.cuda.matmul.allow_tf32 = True

    fusion_types = ["", "add", "relu", "add_relu"]
    shapes = [
        # alexnet
        ([128, 9216], [9216, 4096]),
        # ([128, 4096], [4096, 4096]),
        # ([128, 4096], [4096, 1000]),
        # # BERT
        # ([2048, 768], [768, 768]),
        # ([2048, 768], [768, 3072]),
        # ([2048, 3072], [3072, 768]),
        # # hf_GPT2
        # ([1024, 768], [768, 768]),
        # ([1024, 768], [768, 3072]),
        # ([1024, 3072], [3072, 768]),
        # ([1024, 768], [768, 2304]),
    ]
    for id, shape in enumerate(shapes):
        bench(shape)

    # p = PrettyTable()
    # field_names = ["layer"]
    # for fusion_type in fusion_types:
    #     if fusion_type == "":
    #         field_names.append("torch mm")
    #         field_names.append("triton mm")
    #     else:
    #         field_names.append(f"torch mm+{fusion_type}")
    #         field_names.append(f"triton mm+{fusion_type}")
    #
    # p.field_names = field_names
    # p.float_format = ".3"
    # for id, shape in enumerate(shapes):
    #     bench(shape, id, p, fusion_types)
    #
    # print(p)
    #


if __name__ == "__main__":
    app.run(main)
