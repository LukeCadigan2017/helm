#./test_run_all.ksh meta-llama/Llama-3.1-8B-Instruct 2 1
eval "$(conda shell.bash hook)"
conda activate crfm-helm2

# #model beam_num num_eval
#MODEL=prometheus-eval/prometheus-13b-v1.0
#MODEL=anthropic/claude-v1.3
# MODEL=stas/tiny-random-llama-2
#MODEL=PKU-ONELab/Themis
#MODEL=allenai/OLMo-2-1124-13B-Instruct
#MODEL=meta-llama/Llama-3.1-8B-Instruct
#MODEL=simple/model1
export MODEL=distilbert/distilgpt2
export TASK=wmt
export EVAL_INSTANCES=4
export NUM_BEAMS_LIST=1
export NUM_THREADS=8
export NUM_RETURN_SEQUENCES=4
# export SNELLIUS_METRICS="example_comet"

export SUITE="sample_return_${NUM_RETURN_SEQUENCES}"
./test_run_all.ksh $TASK $MODEL $NUM_BEAMS_LIST $EVAL_INSTANCES $NUM_THREADS $SUITE

#NUM_BEAMS=15
#./test_run_all.ksh $TASK $MODEL $NUM_BEAMS $EVAL_INSTANCES
