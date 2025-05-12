. ./setup_env.ksh

TASK=$1
MODEL=$2
NUM_BEAMS_LIST=$3
EVAL_INSTANCES=$4
NUM_THREADS=$5
NUM_RETURN_SEQUENCES=$6

if [ "$#" -ne 6 ]; then
    echo params num is "$#"
    echo "startup job Usage: $0 <TASK> <MODEL> <NUM_BEAMS_LIST> <EVAL_INSTANCES> <NUM_THREADS> <NUM_RETURN_SEQUENCES>"
    exit 1
fi

echo -e "\n\n\n" 
echo ./test_run_all.ksh $TASK $MODEL $NUM_BEAMS_LIST $EVAL_INSTANCES $NUM_THREADS $NUM_RETURN_SEQUENCES
./test_run_all.ksh $TASK $MODEL $NUM_BEAMS_LIST $EVAL_INSTANCES $NUM_THREADS $NUM_RETURN_SEQUENCES
