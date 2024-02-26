import torch


# @torchdynamo.optimize("inductor", nopython=True)
def inductor_triton_bmm(a, b):
    return torch.bmm(a, b)


def test_total_time(shapes):
    device = torch.device("cuda:0")

    for i in range(len(shapes)):
        a_shape, b_shape = shapes[i]
        print(a_shape, "x", b_shape, end="; ")
        a = torch.randn(a_shape, dtype=torch.float16).to(device)
        b = torch.randn(b_shape, dtype=a.dtype).to(device)

        # https://pytorch.org/docs/stable/generated/torch.compile.html#torch.compile
        fn = torch.compile(
            inductor_triton_bmm,
            backend='inductor',
            mode='max-autotune-no-cudagraphs',
        )
        fn(a, b)


if __name__ == "__main__":
    shapes = [
        # BERT (all)
        # ([192, 128, 64], [192, 64, 128]),
        # ([192, 128, 128], [192, 128, 64]),
        # hf_GPT2 (all)
        ([12, 1024, 1024], [12, 1024, 64]),
        # ([12, 1024, 64], [12, 64, 1024]),
        # # hf_Albert (all)
        # ([12, 512, 64], [12, 64, 512]),
        # ([12, 512, 512], [12, 512, 64]),
    ]

    test_total_time(shapes)
