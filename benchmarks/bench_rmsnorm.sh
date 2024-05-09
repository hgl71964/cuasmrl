#! /bin/bash

workloads=(
        128
        512
     )

Hs=(
        4
        64
     )

for workload in "${workloads[@]}"; do
        for h in "${Hs[@]}"; do
                echo
                echo "workload ${workload}; "
                echo
                python3 benchmarks/rmsnorm.py \
                        --Z 1 \
                        --H $h \
                        --wl $workload  \
                        --dh 128 
                sleep 3
        done
done