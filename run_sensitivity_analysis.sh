#!/bin/bash

for dataset in ams12 archie11 jackson17 taipei13 venice20 rialto20
do
    for alpha in 0.8
    do
        for segments in 5
        do
            echo "${dataset} ${alpha} ${segments}"
            python simulator.py configs/final-dynamic-configs/${dataset}-simple-infer-strata-no-predicate-mean-defensive.json --num-trials 10000 --num-processes 48 --alpha ${alpha} --segments ${segments} --results-dir results-${dataset}-${alpha}-${segments}
        done
    done
done

