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
    echo -e "\n\n\n\n\n\n\n\n\n\n\n\n "
    echo "-------------------------------------------------------------------------------------------------"
}

#################### SETTINGS ####################

#for beam_nums:
#SUITE=eval_$EVAL_INSTANCES

#for return:
#SUITE_NAME=sample_${EVAL_INSTANCES}_instances 

#./snellius_copies/helm_output/sample_return_eval_20/sample_return_eval_20/wmt_14_language_pair_de_en_/meta_llama_Llama_3.1_8B_Instruct/1_beams/runs/sample_return_eval_20/generation_summary_metrics.json


TASK_ENV=$1
MODEL=$2
NUM_BEAMS_LIST=$3
EVAL_INSTANCES=$4
NUM_THREADS=$5
NUM_RETURN_SEQUENCES=$6
#defaults
#NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:=1}"
DISABLE_CACHE="${DISABLE_CACHE:=true}"
RUN_MODEL="${RUN_MODEL:=true}"
# RUN_MODEL="${RUN_MODEL:=true}"
# POST_INSTANCE_METRICS="${POST_INSTANCE_METRICS:=no_metrics}"
# POST_EXAMPLE_METRICS="${POST_EXAMPLE_METRICS:=no_metrics}"

#<TASK> <MODEL> <NUM_BEAMS_LIST> <EVAL_INSTANCES> <NUM_THREADS> <SUITE>
DEFAULT_SUITE=sample_${NUM_RETURN_SEQUENCES}_eval_${EVAL_INSTANCES}
if [ ! -z "$TOP_P" ] ;then
    DEFAULT_SUITE="${DEFAULT_SUITE}_top_p_${TOP_P}"
fi        

if [ ! -z "$TOP_K" ] ;then
    DEFAULT_SUITE="${DEFAULT_SUITE}_top_k_${TOP_K}"
fi



SUITE="${SUITE:=$DEFAULT_SUITE}"


echo SUITE IS $SUITE
echo TASK_ENV is $TASK_ENV
echo MODEL is $MODEL
echo NUM_BEAMS_LIST is $NUM_BEAMS_LIST
echo EVAL_INSTANCES is $EVAL_INSTANCES
echo NUM_RETURN_SEQUENCES is $NUM_RETURN_SEQUENCES
echo DISABLE_CACHE is $DISABLE_CACHE
echo SNELLIUS_METRICS is $SNELLIUS_METRICS
echo RUN_MODEL is $RUN_MODEL
echo TOP_P is $TOP_P
echo TOP_K is $TOP_K
echo "$# is $#"


if ! ([ $NUM_RETURN_SEQUENCES -ge 0 ] || [ $NUM_RETURN_SEQUENCES -lt 0 ]); then
    echo "Please pass an integer but not a string for evaluation."
    exit 1
fi 2>/dev/null


if [ "$#" -ne 6 ]; then
    echo params num is "$#"
    echo "Usage: $0 <TASK> <MODEL> <NUM_BEAMS_LIST> <EVAL_INSTANCES> <NUM_THREADS> <NUM_RETURN_SEQUENCES>"
    exit 1
fi

. $TASK_ENV.env

echo TASK_NAMES IS $TASK_NAMES


# cat ./test_run_all.ksh

HELM_OUTPUT_DIR="${HELM_OUTPUT_DIR:=helm_output}"
SUITE_OUTPUT_DIR=${HELM_OUTPUT_DIR}/${SUITE}
mkdir -p $SUITE_OUTPUT_DIR
OUTPUT_CSV=$SUITE_OUTPUT_DIR/metrics_csv.txt

# $OUTPUT_DIR=$(./create_output_directory.ksh eval_100 wmt meta-llama/Llama-3.2-1B-Instruct 2)

echo $OUTPUT_DIR

# #do everything


for TASK_NAME in $TASK_NAMES; do
    for NUM_BEAMS in $NUM_BEAMS_LIST; do
        echo_space

        echo "TASK_NAMES IS $TASK_NAMES"
        echo "TASK_NAME  IS $TASK_NAME"
        #get run entry and output file names

        RUN_ENTRY=$TASK_NAME

        if [ ! -z "$NUM_RETURN_SEQUENCES" ] ;then
            RUN_ENTRY="${RUN_ENTRY}num_return_sequences=${NUM_RETURN_SEQUENCES},"
        fi

        if [ ! -z "$NUM_BEAMS" ] ;then
            RUN_ENTRY="${RUN_ENTRY}num_beams=${NUM_BEAMS},"
        fi
        
        if [ ! -z "$EOS_TYPE" ] ;then
            RUN_ENTRY="${RUN_ENTRY}eos_type=${EOS_TYPE},"
        fi

        if [ ! -z "$TOP_P" ] ;then
            RUN_ENTRY="${RUN_ENTRY}top_p=${TOP_P},"
        fi        
        
        if [ ! -z "$TOP_K" ] ;then
            RUN_ENTRY="${RUN_ENTRY}top_k=${TOP_K},"
        fi

        if [ ! -z "$BATCH_SIZE" ] ;then
            RUN_ENTRY="${RUN_ENTRY}batch_size=${BATCH_SIZE},"
        fi
        if [ ! -z "$TEMPERATURE" ] ;then
            RUN_ENTRY="${RUN_ENTRY}temperature=${TEMPERATURE},"
        fi
        OUTPUT_PATH="$(./get_output_dir.ksh $SUITE_OUTPUT_DIR $TASK_NAME $MODEL $NUM_BEAMS)"
        TRUE_OUTPUT_PATH=${OUTPUT_PATH}/runs/${SUITE}

        RUN_ENTRY=${RUN_ENTRY}model=${MODEL},follow_format_instructions=instruct
        # if [ -d "$OUTPUT_PATH" ]; then
        #     echo Cannot run! Directory $OUTPUT_PATH already exists
        #     echo $OUTPUT_PATH
        #     ls $OUTPUT_PATH
        #     exit 1 
        # fi
        
        mkdir -p $OUTPUT_PATH
        RUN_PATH=${OUTPUT_PATH}/runs/$SUITE
        STATS_FILE=${RUN_PATH}/stats.json
        
        if [ "$RUN_MODEL" = true ] ; then
            echo helm-run --run-entries $RUN_ENTRY --num-train-trials $NUM_TRAIN_TRIALS --max-eval-instances $EVAL_INSTANCES \
                -o $OUTPUT_PATH --suite $SUITE --num-threads $NUM_THREADS --cache-instances

            if [ "$DISABLE_CACHE" = true ] ; then
                echo "Disable cache"
                helm-run --run-entries $RUN_ENTRY --num-train-trials $NUM_TRAIN_TRIALS --max-eval-instances $EVAL_INSTANCES \
                    -o $OUTPUT_PATH --suite $SUITE  --num-threads $NUM_THREADS --disable-cache --cache-instances
            else
                echo "Do not disable cache"
                helm-run --run-entries $RUN_ENTRY --num-train-trials $NUM_TRAIN_TRIALS --max-eval-instances $EVAL_INSTANCES \
                    -o $OUTPUT_PATH --suite $SUITE  --num-threads $NUM_THREADS --cache-instances
            fi
        fi

        echo STATS_FILE is $STATS_FILE

        #process default stats
        # for DEFAULT_METRIC in $DEFAULT_METRICS; do
        #     echo python process_stats.py --model $MODEL --task  $TASK --num_beams $NUM_BEAMS  --metric $DEFAULT_METRIC \
        #             --stats_file $STATS_FILE --output_csv $OUTPUT_CSV
        #     python process_stats.py --model $MODEL --task  $TASK --num_beams $NUM_BEAMS  --metric $DEFAULT_METRIC \
        #             --stats_file $STATS_FILE --output_csv $OUTPUT_CSV
        # done    
        
        #add snellius metrics
    

        for SNELLIUS_METRIC in $SNELLIUS_METRICS; do
            echo python append_snellius_metrics.py --num_beams $NUM_BEAMS --model $MODEL --eval_instances $EVAL_INSTANCES --task_name $TASK_NAME \
                --run_path ${RUN_PATH} --metric_name $SNELLIUS_METRIC
            python append_snellius_metrics.py --num_beams $NUM_BEAMS --model $MODEL --eval_instances $EVAL_INSTANCES --task_name $TASK_NAME \
                --run_path ${RUN_PATH} --metric_name $SNELLIUS_METRIC
        done

    done
done
# echo -e "\n\n\n\n\n"
# echo "NOTE: USING SLOW TOKENIZER"
# echo "NOTE: USING SLOW TOKENIZER"
# echo "NOTE: USING SLOW TOKENIZER"
# echo "NOTE: USING SLOW TOKENIZER"
# echo "NOTE: USING SLOW TOKENIZER"
# echo "NOTE: USING SLOW TOKENIZER"
# echo "NOTE: USING SLOW TOKENIZER"
# echo "NOTE: USING SLOW TOKENIZER"

# #     #GSM
# #   {description: "gsm:model=text_code,follow_format_instructions=instruct", priority: 2}

# #   # WMT14
# #   {description: "wmt_14:language_pair=cs-en,model=text,follow_format_instructions=instruct", priority: 2}
# #   {description: "wmt_14:language_pair=de-en,model=text,follow_format_instructions=instruct", priority: 2}
# #   {description: "wmt_14:language_pair=fr-en,model=text,follow_format_instructions=instruct", priority: 2}
# #   {description: "wmt_14:language_pair=hi-en,model=text,follow_format_instructions=instruct", priority: 2}
# #   {description: "wmt_14:language_pair=ru-en,model=text,follow_format_instructions=instruct", priority: 2}
