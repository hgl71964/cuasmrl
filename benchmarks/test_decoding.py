import os
import argparse
from dataclasses import dataclass, field

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
    
    # post-process dest; e.g. [R219+0x4000] -> R219
    processed_dst = None
    if dest is not None:
        if dest.startswith('desc'):
            tmp = dest.replace(']', '').split('[')
            tmp = tmp[-1]  # take last register
            processed_dst = tmp.split('.')[0]
        else:
            dest = dest.strip(']').strip('[')
            dest = dest.split('.')[0]
            dest = dest.split('+')[0]
            processed_dst = dest
    
    # post-process src; e.g. ['desc[UR16][R10.64] -> UR16, R10
    processed_src = []
    for i, word in enumerate(src):
        if word.startswith('desc'):
            w = word.replace(']', '').split('[')
            for r in w[1:]: 
                tmp = r.split('.')[0]  # R10.64 -> R10
                processed_src.append(tmp)
        elif word.startswith('c'):
            processed_src.append(word)
        else:
            tmp = word.strip(']').strip('[')
            tmp = tmp.split('.')[0]  # R10.64 -> R10
            tmp = tmp.split('+')[0]  # R10+0x2000 -> R10
            processed_src.append(tmp)

    return ctrl_code, comment, predicate, opcode, processed_dst, processed_src

def decode_ctrl_code(ctrl_code: str):
    ctrl_code = ctrl_code.split(':')
    assert len(ctrl_code) == 5, f'invalid ctrl code: {ctrl_code}'

    barr = ctrl_code[0][2:]
    waits = []
    for bar in barr:
        if bar != '-':
            waits.append(int(bar))

    read = ctrl_code[1]
    write = ctrl_code[2]
    yield_flag = ctrl_code[3]
    stall_count = ctrl_code[4]
    return waits, read, write, yield_flag, stall_count

def main():

    config = parse_args()

    with open(config.path, "r") as f:
        for i, line in enumerate(f):
            # print(line.rstrip())

            ctrl_code, comment, predicate, opcode, dest, src = decode(line)

            print('line is ', line)
            print('decoding: ')
            print(ctrl_code)
            if ctrl_code is not None:
                # skip label
                print(decode_ctrl_code(ctrl_code))
            print(predicate)
            print(opcode)
            print(dest)
            print(src)
            print()

            if i > 20:
                break

if __name__ == "__main__":
    main()
