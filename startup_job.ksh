git pull origin main

source ~/miniconda3/bin/activate
conda activate crfm-helm2 

source hf.env
huggingface-cli login --token $HF_TOKEN --add-to-git-credential


echo huggingface-cli whoami
huggingface-cli whoami

echo -e "\n\n\n" 
echo ./test_run_all.ksh $TASK_NAME $MODEL $NUM_BEAMS $EVAL_INSTANCES $NUM_THREADS
./test_run_all.ksh $TASK_NAME $MODEL $NUM_BEAMS $EVAL_INSTANCES $NUM_THREADS
