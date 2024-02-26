import sys


def main():
    print("Hello, world!")

    import torch

    device = torch.device("cuda:0")
    buffer = torch.randn((100, 100)).cuda()
    print(
        f"Allocated CUDA memory: {torch.cuda.memory_allocated(device) / (1024 ** 2)} MB"
    )
    torch.randn((100, 100), out=buffer)
    print(
        f"Allocated CUDA memory: {torch.cuda.memory_allocated(device) / (1024 ** 2)} MB"
    )
    more = torch.randn((100, 100), device=device)
    print(
        f"Allocated CUDA memory: {torch.cuda.memory_allocated(device) / (1024 ** 2)} MB"
    )

    sys.exit(1)  # Exit the script with an error code


if __name__ == "__main__":
    main()
