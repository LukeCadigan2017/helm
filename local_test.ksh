#./test_run_all.ksh meta-llama/Llama-3.1-8B-Instruct 2 1
eval "$(conda shell.bash hook)"
conda activate crfm-helm2

# #model beam_num num_eval
# MODEL=distilbert/distilgpt2
MODEL=meta-llama/Llama-3.1-8B-Instruct
TASK=wmt
EVAL_INSTANCES=1
NUM_BEAMS=2
./test_run_all.ksh $TASK $MODEL $NUM_BEAMS $EVAL_INSTANCES
