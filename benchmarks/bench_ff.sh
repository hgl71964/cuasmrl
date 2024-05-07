#! /bin/bash

workloads=(
        512
        1024
        # 2048
        4096
     )

for workload in "${workloads[@]}"; do
        echo
        echo "workload ${workload}; "
        echo
        python3 benchmarks/ff.py \
                -m 512 \
                -n 512 \
                -k $workload 
        sleep 3
done