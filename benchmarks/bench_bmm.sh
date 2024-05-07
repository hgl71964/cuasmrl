#! /bin/bash

# config
Bs=(
        1
        2
        4
     )

Ks=(
        512
        1024
        2048
        4096
     )

for b in "${Bs[@]}"; do
        for k in "${Ks[@]}"; do
                echo
                echo "workload ${b}; ${k}"
                echo
                python3 benchmarks/batch_matmul.py \
                        -m 512 \
                        -n 512 \
                        -b $b \ 
                        -k $k
                sleep 3
        done
done