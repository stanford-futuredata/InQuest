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

# run InQuest while varying alpha
for dataset in customer-support archie11 jackson17 taipei13 venice20 rialto20
do
    echo "------ Running sensitivity analysis to alpha on dataset ${dataset} ------"
    for alpha in 0.5 0.6 0.7 0.8 0.9
    do
        echo "${dataset} ${alpha} ${segments}"
        python simulator.py configs/final-inquest-configs/${dataset}-inquest-no-predicate-mean-defensive.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --alpha ${alpha} --segments 5 --results-dir results-${dataset}-${alpha}-5
    done
done

# run InQuest while varying the number of segments
for dataset in customer-support archie11 jackson17 taipei13 venice20 rialto20
do
    echo "------ Running sensitivity analysis to T on dataset ${dataset} ------"
    for segments in 4 5 6 7 8
    do
        echo "${dataset} ${alpha} ${segments}"
        python simulator.py configs/final-inquest-configs/${dataset}-inquest-no-predicate-mean-defensive.json --trials-per-oracle-limit ${TRIALS_PER_ORACLE_LIMIT} --num-processes ${NUM_PROCESSES} --alpha 0.8 --segments ${segments} --results-dir results-${dataset}-0.8-${segments}
    done
done

# can run the above on queries w/a predicate by swapping "no-predicate" --> "predicate-gt0" in config filenames
