#! /bin/bash

# config
seeds=100
workloads=(
        11
        12
        13
        14
        15
     )

for seed in $( seq 1 $seeds ); do
        for workload in "${workloads[@]}"; do
                echo
                echo "workload ${workload}; seed ${seed}: "
                echo
                python benchmarks/11-grouped-gemm.py \
                        --seed $seed \
                        --wl $workload
        done
done
