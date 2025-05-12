#./test_run_all.ksh meta-llama/Llama-3.1-8B-Instruct 2 1
eval "$(conda shell.bash hook)"
conda activate crfm-helm2

# #model beam_num num_eval
# MODEL=distilbert/distilgpt2
export MODEL=stas/tiny-random-llama-2
# export MODEL=simple/model1
#MODEL=meta-llama/Llama-3.1-8B-Instruct
export TASK=instruct
export EVAL_INSTANCES=1
export NUM_BEAMS=1
export NUM_THREADS=2
export NUM_RETURN_SEQUENCES=2
#export SUITE=instruct_${EVAL_INSTANCES}_evals

# export SNELLIUS_METRICS=example_themis
echo SNELLIUS METRICS IS $SNELLIUS_METRICS

echo "NUM_BEAMS IS $NUM_BEAMS"
echo ./test_run_all.ksh $TASK $MODEL $NUM_BEAMS $EVAL_INSTANCES $NUM_THREADS $SUITE
. ./test_run_all.ksh $TASK $MODEL $NUM_BEAMS $EVAL_INSTANCES $NUM_THREADS $SUITE


#NUM_BEAMS=15
#./test_run_all.ksh $TASK $MODEL $NUM_BEAMS $EVAL_INSTANCES

#helm_output/instruct_1_evals/open_assistant_language_en_num_respondents_1_/simple_model1/2_beams/runs/instruct_1_evals/generation_summary.json
#helm_output/instruct_1_evals/open_assistant_language_en_num_respondents_1_/simple_model1/2_beams/runs/instruct_1_evals/generation_summary.json
