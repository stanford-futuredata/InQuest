#!/bin/bash

for dataset in ams12 archie11 jackson17 taipei13 venice20 rialto20
do
    for segments in 4
    do
        echo "${dataset} uniform"
        python simulator.py configs/final-uniform-configs/${dataset}-uniform-no-predicate-mean.json --num-trials 10000 --num-processes 48 --results-dir results-uniform-${dataset}
        echo "${dataset} static ${segments}"
        python simulator.py configs/final-static-configs/${dataset}-static-no-predicate-mean.json --num-trials 10000 --num-processes 48 --segments ${segments} --results-dir results-static-${dataset}-${segments}
    done
done
