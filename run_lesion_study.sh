#!/bin/bash

for dataset in ams12 archie11 jackson17 taipei13 venice20 rialto20
do
    for alpha in 0.8
    do
        for segments in 5
        do
            echo "${dataset} ${alpha} ${segments} fixed strata dyn. alloc. -- no pred"
            python simulator.py configs/final-fixed-strata-dynamic-alloc/${dataset}-no-predicate-mean-defensive.json --num-trials 10000 --num-processes 48 --alpha ${alpha} --segments ${segments} --results-dir results-${dataset}-${alpha}-${segments}-fixed-strata-no-pred
            echo "${dataset} ${alpha} ${segments} dyn. strata fixed alloc. -- no pred"
            python simulator.py configs/final-dynamic-strata-fixed-alloc/${dataset}-no-predicate-mean-defensive.json --num-trials 10000 --num-processes 48 --alpha ${alpha} --segments ${segments} --results-dir results-${dataset}-${alpha}-${segments}-fixed-alloc-no-pred
            echo "${dataset} ${alpha} ${segments} fixed strata dyn. alloc. -- pred gt0"
            python simulator.py configs/final-fixed-strata-dynamic-alloc/${dataset}-predicate-gt0-mean-defensive.json --num-trials 10000 --num-processes 48 --alpha ${alpha} --segments ${segments} --results-dir results-${dataset}-${alpha}-${segments}-fixed-strata-pred-gt0
            echo "${dataset} ${alpha} ${segments} dyn. strata fixed alloc. -- pred gt0"
            python simulator.py configs/final-dynamic-strata-fixed-alloc/${dataset}-predicate-gt0-mean-defensive.json --num-trials 10000 --num-processes 48 --alpha ${alpha} --segments ${segments} --results-dir results-${dataset}-${alpha}-${segments}-fixed-alloc-pred-gt0
        done
    done
done

