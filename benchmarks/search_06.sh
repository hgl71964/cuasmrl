#! /bin/bash

# config
start=${1:-1}
seeds=${2:-100}
workloads=(
        16384
        8192
        4096
        2048
        1024
        512
        # 32768
     )

Zs=(
        1
        16
     )

Hs=(
        4
        16
        64
     )

HEADs=(
        32
        64
        128
        # 256
     )

for seed in $( seq $start $seeds ); do
        for Z in "${Zs[@]}"; do
                for H in "${Hs[@]}"; do
                        for HEAD in "${HEADs[@]}"; do
                                for workload in "${workloads[@]}"; do
                                        echo
                                        echo "workload ${Z}_${H}_${workload}_${HEAD}; seed ${seed}: "
                                        echo
                                        python benchmarks/06-fused-attention.py \
                                                --seed $seed \
                                                --Z $Z \
                                                --H $H \
                                                --D_HEAD $HEAD \
                                                --wl $workload
                                        sleep 3
                                done
                        done
                done
        done
done
