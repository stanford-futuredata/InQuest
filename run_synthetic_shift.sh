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

for num_shifts in 1 2 3 4 5
do
    for vec in {0..19}
    do
        echo "--- uniform num_shifts ${num_shifts} vec ${vec} ---"
        python simulator.py configs/final-synthetic-uniform-configs/config_num_shifts_${num_shifts}_vec_${vec}.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --results-dir results-uniform-num-shifts-${num_shifts}-vec-${vec} --single-budget True
        echo "--- static num_shifts ${num_shifts} vec ${vec} ---"
        python simulator.py configs/final-synthetic-static-configs/config_num_shifts_${num_shifts}_vec_${vec}.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --segments 5 --results-dir results-static-num-shifts-${num_shifts}-vec-${vec} --single-budget True
        echo "--- InQuest num_shifts ${num_shifts} vec ${vec} ---"
        python simulator.py configs/final-synthetic-inquest-configs/config_num_shifts_${num_shifts}_vec_${vec}.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --segments 5 --alpha 0.8 --results-dir results-inquest-num-shifts-${num_shifts}-vec-${vec} --single-budget True
    done
done
