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
        -p|--num-processes)
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

# run baselines and inquest over each dataset
for dataset in customer-support archie11 jackson17 taipei13 venice20 rialto20
do
    echo "------ Running baselines + InQuest on dataset ${dataset} ------"
    echo "${dataset} uniform"
    python simulator.py configs/final-uniform-configs/${dataset}-uniform-no-predicate-mean.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --results-dir results-uniform-${dataset}
    echo "${dataset} static"
    python simulator.py configs/final-static-configs/${dataset}-static-no-predicate-mean.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --segments 4 --results-dir results-static-${dataset}
    echo "${dataset} inquest"
    python simulator.py configs/final-inquest-configs/${dataset}-inquest-no-predicate-mean.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --segments 5 --alpha 0.8 --results-dir results-inquest-${dataset}
done
