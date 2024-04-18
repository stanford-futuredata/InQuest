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

# run baselines and InQuest over each dataset for queries without a predicate
for dataset in customer-support archie11 jackson17 taipei13 venice20 rialto20
do
    echo "------ Running baselines + InQuest on dataset ${dataset} ------"
    echo "--- ${dataset} uniform ---"
    python simulator.py configs/final-uniform-configs/${dataset}-uniform-no-predicate-mean.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --segments 5 --results-dir results-uniform-${dataset}-no-pred
    echo "--- ${dataset} static ---"
    python simulator.py configs/final-static-configs/${dataset}-static-no-predicate-mean.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --segments 5 --results-dir results-static-${dataset}-no-pred
    echo "--- ${dataset} InQuest ---"
    python simulator.py configs/final-inquest-configs/${dataset}-inquest-no-predicate-mean.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --segments 5 --alpha 0.8 --results-dir results-inquest-${dataset}-no-pred
done

# run baselines and InQuest over each dataset for queries with a predicate
for dataset in customer-support archie11 jackson17 taipei13 venice20 rialto20
do
    echo "------ Running baselines + InQuest on dataset ${dataset} ------"
    echo "--- ${dataset} uniform ---"
    python simulator.py configs/final-uniform-configs/${dataset}-uniform-predicate-gt0-mean.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --segments 5 --results-dir results-uniform-${dataset}-pred-gt0
    echo "--- ${dataset} static ---"
    python simulator.py configs/final-static-configs/${dataset}-static-predicate-gt0-mean.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --segments 5 --results-dir results-static-${dataset}-pred-gt0
    echo "--- ${dataset} InQuest ---"
    python simulator.py configs/final-inquest-configs/${dataset}-inquest-predicate-gt0-mean.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --segments 5 --alpha 0.8 --results-dir results-inquest-${dataset}-pred-gt0
done

