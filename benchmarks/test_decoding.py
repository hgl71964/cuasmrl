import os
import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch

import triton
import triton.language as tl

import random
import numpy as np

# yapf: disable
@dataclass
class Config:
    path: str = "data"


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="???")
    parser.add_argument('-p', "--default_out_path", type=str, dest="path", default="data")
    args = parser.parse_args()
    config = Config(**vars(args))
    return config

def decode(line: str):
    line = line.strip('\n')
    line = line.split(' ')
    n = len(line)

    ctrl_code = None
    predicate = None
    comment = None
    opcode = None
    dest = None
    src = []

    idx = -1
    for i in range(0, n):
        if line[i] != '':
            idx = i
            ctrl_code = line[i]
            break
    assert idx > -1, f'no ctrl: {line}'

    if ctrl_code.startswith('.'):
        # labels
        return None, None, None, None, None, None

    for i in range(idx + 1, n):
        if line[i] != '':
            idx = i
            comment = line[i]
            break

    for i in range(idx + 1, n):
        if line[i] != '':

            if line[i][0] == '@':
                predicate = line[i]
            else:
                opcode = line[i]

            idx = i
            break

    if opcode is None:
        for i in range(idx + 1, n):
            if line[i] != '':
                opcode = line[i]
                idx = i
                break

    for i in range(idx + 1, n):
        if line[i] != '':
            dest = line[i].strip(',')
            idx = i
            break

    if dest == ';':
        # LDGDEPBAR inst
        dest = None

    for i in range(idx + 1, n):
        if line[i] == ';':
            break

        if line[i] != '':
            src.append(line[i].strip(','))

    return ctrl_code, comment, predicate, opcode, dest, src

def decode_ctrl_code(ctrl_code: str):
    ctrl_code = ctrl_code.split(':')
    assert len(ctrl_code) == 5, f'invalid ctrl code: {ctrl_code}'

    barr = ctrl_code[0]
    read = ctrl_code[1]
    write = ctrl_code[2]
    yield_flag = ctrl_code[3]
    stall_count = ctrl_code[4]
    return barr, read, write, yield_flag, stall_count

def main():

    config = parse_args()

    with open(config.path, "r") as f:
        for i, line in enumerate(f):
            # print(line.rstrip())

            ctrl_code, comment, predicate, opcode, dest, src = decode(line)

            print(ctrl_code)
            print(decode_ctrl_code(ctrl_code))
            print(predicate)
            print(opcode)
            print(dest)
            print(src)
            print()

            if i > 5:
                break

if __name__ == "__main__":
    main()
