#./test_run_all.ksh meta-llama/Llama-3.1-8B-Instruct 2 1
#eval "$(conda shell.bash hook)"
conda activate crfm-helm2

# #model beam_num num_eval
MODEL=distilbert/distilgpt2
#export MODEL=simple/model1
#MODEL=meta-llama/Llama-3.1-8B-Instruct
export TASK=instruct
export EVAL_INSTANCES=1
export NUM_BEAMS=2
export NUM_THREADS=2


echo ./test_run_all.ksh $TASK $MODEL $NUM_BEAMS $EVAL_INSTANCES $NUM_THREADS
./test_run_all.ksh $TASK $MODEL $NUM_BEAMS $EVAL_INSTANCES $NUM_THREADS


#NUM_BEAMS=15
#./test_run_all.ksh $TASK $MODEL $NUM_BEAMS $EVAL_INSTANCES
