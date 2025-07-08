#./test_run_all.ksh meta-llama/Llama-3.1-8B-Instruct 2 1
eval "$(conda shell.bash hook)"
conda activate crfm-helm2


# export HF_HOME=/scratch-local/lcadigan/cache/huggingface




source hf.env
huggingface-cli login --token $HF_TOKEN --add-to-git-credential


echo huggingface-cli whoami
huggingface-cli whoami




# #model beam_num num_eval
# export MODEL=allenai/OLMo-2-1124-13B-Instruct
#MODEL=prometheus-eval/prometheus-13b-v1.0
#MODEL=anthropic/claude-v1.3

# MODEL=allenai/OLMo-2-0425-1B-Instruct
# MODEL=allenai/OLMo-2-0325-32B-Instruct
# MODEL=allenai/OLMo-2-1124-7B-Instruct


#MODEL=PKU-ONELab/Themis
#MODEL=allenai/OLMo-2-1124-13B-Instruct

# export MODEL=simple/model1
# export MODEL=allenai/OLMo-2-1124-13B-Instruct
# export MODEL=Qwen/Qwen2.5-7B-Instruct
# export MODEL=meta-llama/Llama-3.2-1B

# export MODEL=meta-llama/Llama-3.1-8B
# export MODEL=stas/tiny-random-llama-2
# export MODEL=sshleifer/tiny-gpt2
export MODEL=distilbert/distilgpt2
# export MODEL=Qwen/Qwen3-0.6B
# export MODEL=meta-llama/Llama-3.2-1B-Instruct

# export MODEL=allenai/OLMo-2-0425-1B
# export MODEL=meta-llama/Llama-3.1-8B-Instruct
# export MODEL=allenai/OLMo-2-1124-7B-RM
# export MODEL=allenai/OLMo-2-1124-13B-RM



######## USE THESE ########

export NUM_THREADS=1    
 
# export TASK=instruct
# export EOS_TYPE=""

export TASK=wmt
# export TASK=instruct
# export TASK=gsm
# export EOS_TYPE="task"

#should be 0-3
# export EVAL_INSTANCES=4
# export FIRST_RUN_INSTANCE=0


#should be 3
#Die Text, der gestern Abend noch verabschiedet worden ist, besteht aus 234 Artikeln.
export EVAL_INSTANCES=2
export FIRST_RUN_INSTANCE=0



export NUM_BEAMS_LIST=1
export NUM_RETURN_SEQUENCES=2


export BATCH_SIZE=1000
# export TOP_P=1
export EXACT_MODE=false

# export TOP_K=2



export INSTRUCT=true
export RUN_MODEL=true
export DISABLE_CACHE=true
export TEMPLATE=true
# export SNELLIUS_METRICS="example_comet"

######## USE THESE ########
# HELM_OUTPUT_DIR=snellius_copies/helm_output




echo ./test_run_all.ksh $TASK $MODEL $NUM_BEAMS_LIST $EVAL_INSTANCES $NUM_THREADS $NUM_RETURN_SEQUENCES
./test_run_all.ksh $TASK $MODEL $NUM_BEAMS_LIST $EVAL_INSTANCES $NUM_THREADS $NUM_RETURN_SEQUENCES

#NUM_BEAMS=15
#./test_run_all.ksh $TASK $MODEL $NUM_BEAMS $EVAL_INSTANCES


