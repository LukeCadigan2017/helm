#!/bin/bash
export SUITE_NAME="tokenizer_test"
export NUM_TRAIN_TRIALS=1
export MAX_EVAL_INSTANCES=10
export MODELS_TO_RUN=openai/gpt2
export NUM_BEAMS=1


echo "Cat the script"

cat ./test_tokenizer.ksh


echo helm-run --run-entries wmt_14:language_pair=cs-en,model=$MODELS_TO_RUN,follow_format_instructions=instruct,num_beams=$NUM_BEAMS --num-train-trials $NUM_TRAIN_TRIALS --max-eval-instances $MAX_EVAL_INSTANCES --suite $SUITE_NAME

helm-run --run-entries wmt_14:language_pair=cs-en,model=$MODELS_TO_RUN,follow_format_instructions=instruct,num_beams=$NUM_BEAMS --num-train-trials $NUM_TRAIN_TRIALS --max-eval-instances $MAX_EVAL_INSTANCES --suite $SUITE_NAME
