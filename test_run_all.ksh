#!/bin/bash



#################### FUNCTIONS ####################
clean_str () {
    CLEAN_STR=$1
    chars="= , : __ - / "
    for char in $chars; do
        CLEAN_STR=${CLEAN_STR//$char/_}
    done
}


echo_space () {
    echo "-------------------------------------------------------------------------------------------------"
    echo -e "\n\n\n\n\n\n\n\n\n\n\n\n "

}

#################### SETTINGS ####################


TASK_NAME=$1
MODEL=$2
NUM_BEAMS_LIST=$3
MAX_EVAL_INSTANCES=$4

echo TASK_ENV is $TASK_ENV
echo MODEL is $MODEL
echo NUM_BEAMS_LIST is $NUM_BEAMS_LIST
echo MAX_EVAL_INSTANCES is $MAX_EVAL_INSTANCES

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <TASK_NAME> <MODEL> <NUM_BEAMS_LIST> <EVAL_INSTANCES>"
    exit 1
fi

. $TASK_NAME.env

cat ./test_run_all.ksh
SUITE=eval_$MAX_EVAL_INSTANCES
SUITE_OUTPUT_DIR=helm_output/${SUITE}
mkdir -p $SUITE_OUTPUT_DIR
OUTPUT_CSV=$SUITE_OUTPUT_DIR/metrics_csv.txt

# $OUTPUT_DIR=$(./create_output_directory.ksh eval_100 wmt meta-llama/Llama-3.2-1B-Instruct 2)

echo $OUTPUT_DIR

# #do everything

for NUM_BEAMS in $NUM_BEAMS_LIST; do
    echo_space

    #get run entry and output file names
    RUN_ENTRY=${TASK}model=${MODEL},follow_format_instructions=instruct,num_beams=$NUM_BEAMS

    OUTPUT_PATH="$(./get_output_dir.ksh $SUITE_OUTPUT_DIR $TASK_NAME $MODEL $NUM_BEAMS)"

    # if [ -d "$OUTPUT_PATH" ]; then
    #     echo Cannot run! Directory $OUTPUT_PATH already exists
    #     echo $OUTPUT_PATH
    #     ls $OUTPUT_PATH
    #     exit 1 
    # fi
    
    mkdir -p $OUTPUT_PATH

    STATS_FILE=$OUTPUT_PATH/runs/$SUITE/stats.json
    
    helm-run --run-entries $RUN_ENTRY --num-train-trials $NUM_TRAIN_TRIALS --max-eval-instances $MAX_EVAL_INSTANCES \
        -o $OUTPUT_PATH --suite $SUITE --disable-cache
    echo STATS_FILE is $STATS_FILE

    #process results
    for METRIC in $METRICS; do
        python process_stats.py --model $MODEL --task  $TASK --num_beams $NUM_BEAMS  --metric $METRIC \
                --stats_file $STATS_FILE --output_csv $OUTPUT_CSV
    done

done

# #     #GSM
# #   {description: "gsm:model=text_code,follow_format_instructions=instruct", priority: 2}

# #   # WMT14
# #   {description: "wmt_14:language_pair=cs-en,model=text,follow_format_instructions=instruct", priority: 2}
# #   {description: "wmt_14:language_pair=de-en,model=text,follow_format_instructions=instruct", priority: 2}
# #   {description: "wmt_14:language_pair=fr-en,model=text,follow_format_instructions=instruct", priority: 2}
# #   {description: "wmt_14:language_pair=hi-en,model=text,follow_format_instructions=instruct", priority: 2}
# #   {description: "wmt_14:language_pair=ru-en,model=text,follow_format_instructions=instruct", priority: 2}
