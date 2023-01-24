#!/bin/bash

# initialize variables with default values
TRIALS_PER_ORACLE_LIMIT=-1
NUM_PROCESSES=48

# parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--trials-per-oracle-limit)
        TRIALS_PER_ORACLE_LIMIT="$2"
        shift # past argument
        shift # past value
        ;;
        -n|--num-processes)
        NUM_PROCESSES="$2"
        shift # past argument
        shift # past value
        ;;
        -*|--*)
        echo "ERROR: unknown option"
        exit 1
        ;;
        *)
        ;;
    esac
done

for dataset in customer-support archie11 jackson17 taipei13 venice20 rialto20
do
    echo "------ Running lesion experiments on dataset ${dataset} ------"
    echo "--- fixed strata dyn. alloc. -- no pred ${dataset} ---"
    python simulator.py configs/final-fixed-strata-dynamic-alloc/${dataset}-no-predicate-mean.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --alpha 0.8 --segments 5 --results-dir results-${dataset}-0.8-5-fixed-strata-no-pred
    echo "--- dyn. strata fixed alloc. -- no pred ${dataset} ---"
    python simulator.py configs/final-dynamic-strata-fixed-alloc/${dataset}-no-predicate-mean.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --alpha 0.8 --segments 5 --results-dir results-${dataset}-0.8-5-fixed-alloc-no-pred
    # echo "fixed strata dyn. alloc. -- pred gt0 ${dataset}"
    # python simulator.py configs/final-fixed-strata-dynamic-alloc/${dataset}-predicate-gt0-mean.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --alpha 0.8 --segments 5 --results-dir results-${dataset}-0.8-5-fixed-strata-pred-gt0
    # echo "dyn. strata fixed alloc. -- pred gt0 ${dataset}"
    # python simulator.py configs/final-dynamic-strata-fixed-alloc/${dataset}-predicate-gt0-mean.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --alpha 0.8 --segments 5 --results-dir results-${dataset}-0.8-5-fixed-alloc-pred-gt0
done

