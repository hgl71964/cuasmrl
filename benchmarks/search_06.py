import sys
import subprocess
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description="???")

    parser.add_argument("--default_out_path", type=str, default="data")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n_tests", type=int, default=5)
    parser.add_argument("--load", type=str)
    parser.add_argument("--bench", type=int, default=0)

    parser.add_argument("--Z", type=int, default=1)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--wl", type=int, default=16384)
    parser.add_argument("--dh", type=int, dest="dh", default=64)

    parser.add_argument("-t", "--train", type=int, dest="train", default=1)
    parser.add_argument("-l", "--log", type=int, dest="log", default=1)
    parser.add_argument("--verbose", type=int, default=0)

    parser.add_argument("--env_id", type=str, default='cuasmenv-v0')
    parser.add_argument("--num_iterations", type=int, default=int(1e6))
    parser.add_argument("--minibatch_size", type=int, default=8)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--normalize_reward", type=int, default=0)
    parser.add_argument("--ckpt_freq", type=int, default=10)

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
    return args 

def construct_command_line_args(args):
    args_dict = vars(args)
    command_line_args = []
    for arg, value in args_dict.items():

        # skip args
        if value is None:
            continue

        command_line_args.append(f"--{arg}")
        command_line_args.append(str(value))
    return command_line_args


def main():
    args = parse_args()
    kernel = "06-fused-attention"
    command = ["python3", "benchmarks/06-fused-attention.py"]

    # args
    command_line_args = construct_command_line_args(args)
    command += command_line_args

    # run
    return_code = 1
    while return_code != 0:
        result = subprocess.run(command, capture_output=True, text=True)
        return_code = result.returncode
        if return_code == 0:
            print(f"{kernel} finished successfully.")
        else:
            print(f"{kernel} exited with code {return_code}, restarting...")
            time.sleep(3)

if __name__ == "__main__":
    main()