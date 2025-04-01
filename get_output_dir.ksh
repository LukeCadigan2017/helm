#!/bin/bash



#################### FUNCTIONS ####################
clean_str () {
    CLEAN_STR=$1
    chars="= , : __ - / "
    for char in $chars; do
        CLEAN_STR=${CLEAN_STR//$char/_}
    done
}

#EVAL_INSTANCES contained in SUITE
BASE_DIR=$1
TASK_NAME=$2
MODEL=$3
NUM_BEAMS=$4

BEAM_NAME=${NUM_BEAMS}_beams

# ./test_run_all.ksh wmt meta-llama/Llama-3.2-1B-Instruct 2 600 "bleu_4 comet"


if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <BASE_DIR> <TASK_NAME> <MODEL> <NUM_BEAMS>"
    exit 1
fi


clean_str $MODEL
CLEAN_MODEL=$CLEAN_STR

clean_str $TASK_NAME
CLEAN_TASK=$CLEAN_STR

clean_str $BEAM_NAME
CLEAN_BEAM_NAME=$CLEAN_STR

BASE_OUTPUT_DIR=${BASE_DIR}/${CLEAN_TASK}/${CLEAN_MODEL}/${CLEAN_BEAM_NAME}

echo $BASE_OUTPUT_DIR