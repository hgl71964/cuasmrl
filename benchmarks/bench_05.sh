#! /bin/bash

# config
seeds=100
workloads=(
        512
        1024
        2048
        4096
        8192
        16384
        32768
     )

for seed in $( seq 1 $seeds ); do
        for workload in "${workloads[@]}"; do
                echo
                echo "workload ${workload}; seed ${seed}: "
                echo
                python benchmarks/05-layer-norm.py \
                        --seed $seed \
                        --wl $workload
        done
done
