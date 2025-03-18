#!/bin/bash



#################### FUNCTIONS ####################
clean_str () {
    SUITE_NAME=$1
    chars="/ = , : __ -"
    for char in $chars; do
        SUITE_NAME=${SUITE_NAME//$char/_}
    done
}


echo_space () {
    echo "-------------------------------------------------------------------------------------------------"
    echo -e "\n\n\n\n\n\n\n\n\n\n\n\n "

}

#################### SETTINGS ####################

#task stuff
NUM_TRAIN_TRIALS=1
TASK=wmt_14:language_pair=de-en
METRICS="bleu_4 comet"

#other configs

NUM_BEAMS_LIST="2"

MAX_EVAL_INSTANCES=10
#MODELS="meta-llama/Llama-3.1-8B"
MODELS="meta-llama/Llama-3.2-1B-Instruct"
# MODELS="stas/tiny-random-llama-2"



#base stuff
SUITE_BASE="run_all_evalnum_${MAX_EVAL_INSTANCES}"
BASE_OUTPUT_DIR=benchmark_output
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${SUITE_BASE}"
OUTPUT_CSV=$OUTPUT_DIR/metrics_csv.txt

#create files
mkdir -p $BASE_OUTPUT_DIR
mkdir -p $OUTPUT_DIR
touch $OUTPUT_CSV

#book-keeping
GEN_OUTPUT_FILES=""
cat ./test_run_all.ksh


#do everything
for MODEL in $MODELS; do
    for NUM_BEAMS in $NUM_BEAMS_LIST; do
        echo_space

        #get run entry and output file names
        RUN_ENTRY=${TASK},model=${MODEL},follow_format_instructions=instruct,num_beams=$NUM_BEAMS
        
        #sets SUITE_NAME
        clean_str "${SUITE_BASE}_${RUN_ENTRY}"
            
        GEN_OUTPUT_FILE="${OUTPUT_DIR}/generated_${SUITE_NAME}.json"
        RUN_ENTRY=${RUN_ENTRY},generated_output_file=${GEN_OUTPUT_FILE}

        #book-keeping
        touch ${GEN_OUTPUT_FILE}
        GEN_OUTPUT_FILES="${GEN_OUTPUT_FILES} ${GEN_OUTPUT_FILE}"     

        
        helm-run --run-entries $RUN_ENTRY --num-train-trials $NUM_TRAIN_TRIALS --max-eval-instances $MAX_EVAL_INSTANCES --suite $SUITE_NAME --disable-cache
        

        #process results
        for METRIC in $METRICS; do
            python process_results.py --model $MODEL --task  $TASK --num_beams $NUM_BEAMS  --metric $METRIC \
                --suite_name $SUITE_NAME --output_csv $OUTPUT_CSV
        done
        # python process_generation.py --gen_output_json $GEN_OUTPUT_FILE

    done

done

#echo_space

#print out relevant files
# for GEN_OUTPUT_FILE in $GEN_OUTPUT_FILES; do
#     echo "GEN_OUTPUT_FILE: ${GEN_OUTPUT_FILE}"
# done
#echo OUTPUT_CSV is $OUTPUT_CSV

#USE THIS ONE 
#helm-run --run-entries gsm_iom:model=stas/tiny-random-llama-2,num_beams=2  --suite my-suite --max-eval-instances 2 --disable-cache
# beam_1739389179


#     #GSM
#   {description: "gsm:model=text_code,follow_format_instructions=instruct", priority: 2}

#   # WMT14
#   {description: "wmt_14:language_pair=cs-en,model=text,follow_format_instructions=instruct", priority: 2}
#   {description: "wmt_14:language_pair=de-en,model=text,follow_format_instructions=instruct", priority: 2}
#   {description: "wmt_14:language_pair=fr-en,model=text,follow_format_instructions=instruct", priority: 2}
#   {description: "wmt_14:language_pair=hi-en,model=text,follow_format_instructions=instruct", priority: 2}
#   {description: "wmt_14:language_pair=ru-en,model=text,follow_format_instructions=instruct", priority: 2}
