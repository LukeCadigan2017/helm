git pull origin main

source ~/miniconda3/bin/activate
conda activate crfm-helm2 

source hf.env
huggingface-cli login --token $HF_TOKEN --add-to-git-credential


echo huggingface-cli whoami
huggingface-cli whoami

TASK=$1
MODEL=$2
NUM_BEAMS_LIST=$3
EVAL_INSTANCES=$4
NUM_THREADS=$5
SUITE=$6

if [ "$#" -lt 6 ]; then
    echo "Usage: $0 <TASK> <MODEL> <NUM_BEAMS_LIST> <EVAL_INSTANCES> <NUM_THREADS> <SUITE>"
    exit 1
fi

echo -e "\n\n\n" 
echo ./test_run_all.ksh $TASK $MODEL $NUM_BEAMS_LIST $EVAL_INSTANCES $NUM_THREADS $SUITE
./test_run_all.ksh $TASK $MODEL $NUM_BEAMS_LIST $EVAL_INSTANCES $NUM_THREADS $SUITE
