import torch
import subprocess


def get_gpu_name():
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            universal_newlines=True)
        gpu_name = output.strip()
        gpu_name = gpu_name.replace(' ', '_')
        return gpu_name
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error: {e}")


def get_gpu_cc():
    if torch.cuda.is_available():
        # device = torch.device("cuda")
        # print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        compute_capability = torch.cuda.get_device_capability(0)
        # print(f"Compute Capability: {compute_capability}")
        return compute_capability  # e.g. (7, 5)
    else:
        raise RuntimeError("No GPU available, using CPU.")
