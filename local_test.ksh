#./test_run_all.ksh meta-llama/Llama-3.1-8B-Instruct 2 1
eval "$(conda shell.bash hook)"
conda activate crfm-helm2

# #model beam_num num_eval
#MODEL=prometheus-eval/prometheus-13b-v1.0
#MODEL=anthropic/claude-v1.3
# MODEL=stas/tiny-random-llama-2
# MODEL=allenai/OLMo-2-0425-1B-Instruct
# MODEL=allenai/OLMo-2-0325-32B-Instruct
# MODEL=allenai/OLMo-2-1124-7B-Instruct


#MODEL=PKU-ONELab/Themis
#MODEL=allenai/OLMo-2-1124-13B-Instruct

#MODEL=simple/model1
#export MODEL=distilbert/distilgpt2

# export MODEL=allenai/OLMo-2-1124-7B-RM
# export MODEL=allenai/OLMo-2-1124-13B-RM



######## USE THESE ########

export NUM_THREADS=8

export TASK=wmt
export EVAL_INSTANCES=1000
export NUM_BEAMS_LIST=8
export NUM_RETURN_SEQUENCES=1

export MODEL=meta-llama/Llama-3.1-8B-Instruct
# export SUITE="full_wmt_1_samples_1000_evals"

export RUN_MODEL=false
export SNELLIUS_METRICS="example_comet"

######## USE THESE ########
# HELM_OUTPUT_DIR=snellius_copies/helm_output




echo ./test_run_all.ksh $TASK $MODEL $NUM_BEAMS_LIST $EVAL_INSTANCES $NUM_THREADS $NUM_RETURN_SEQUENCES
./test_run_all.ksh $TASK $MODEL $NUM_BEAMS_LIST $EVAL_INSTANCES $NUM_THREADS $NUM_RETURN_SEQUENCES

#NUM_BEAMS=15
#./test_run_all.ksh $TASK $MODEL $NUM_BEAMS $EVAL_INSTANCES


