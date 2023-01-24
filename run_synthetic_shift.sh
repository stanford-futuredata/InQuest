#!/bin/bash

for scale in 1 2 3 4 5
do
    for vec in 1 2 3 4 5 6 7 8 9 10
    do
        echo "dynamic scale ${scale} vec ${vec}"
        python simulator.py configs/final-synthetic-dynamic-configs/config_scale_${scale}_vec_${vec}.json --num-trials 1000 --num-processes 48 --results-dir results-dynamic-scale-${scale}-vec-${vec}
        echo "static scale ${scale} vec ${vec}"
        python simulator.py configs/final-synthetic-static-configs/config_scale_${scale}_vec_${vec}.json --num-trials 1000 --num-processes 48 --results-dir results-static-scale-${scale}-vec-${vec}
        echo "uniform scale ${scale} vec ${vec}"
        python simulator.py configs/final-synthetic-uniform-configs/config_scale_${scale}_vec_${vec}.json --num-trials 1000 --num-processes 48 --results-dir results-uniform-scale-${scale}-vec-${vec}
    done
done
