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

for scale in 1 2 3 4 5
do
    for vec in 1 2 3 4 5 6 7 8 9 10
    do
        echo "--- uniform scale ${scale} vec ${vec} ---"
        python simulator.py configs/final-synthetic-uniform-configs/config_scale_${scale}_vec_${vec}.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --results-dir results-uniform-scale-${scale}-vec-${vec}
        echo "--- static scale ${scale} vec ${vec} ---"
        python simulator.py configs/final-synthetic-static-configs/config_scale_${scale}_vec_${vec}.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --segments 4 --results-dir results-static-scale-${scale}-vec-${vec}
        echo "--- InQuest scale ${scale} vec ${vec} ---"
        python simulator.py configs/final-synthetic-inquest-configs/config_scale_${scale}_vec_${vec}.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --segments 5 --alpha 0.8 --results-dir results-inquest-scale-${scale}-vec-${vec}
    done
done
