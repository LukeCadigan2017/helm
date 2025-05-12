. ./setup_env.ksh

TASK=$1
MODEL=$2
NUM_BEAMS_LIST=$3
EVAL_INSTANCES=$4
NUM_THREADS=$5

if [ "$#" -lt 6 ]; then
    echo "startup job Usage: $0 <TASK> <MODEL> <NUM_BEAMS_LIST> <EVAL_INSTANCES> <NUM_THREADS>"
    exit 1
fi

echo -e "\n\n\n" 
echo ./test_run_all.ksh $TASK $MODEL $NUM_BEAMS_LIST $EVAL_INSTANCES $NUM_THREADS 
./test_run_all.ksh $TASK $MODEL $NUM_BEAMS_LIST $EVAL_INSTANCES $NUM_THREADS 

