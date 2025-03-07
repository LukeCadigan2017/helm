#!/bin/bash


#base stuff
SUITE_BASE="test_run_all"
OUTPUT_CSV=${SUITE_BASE}.txt

#task stuff
NUM_TRAIN_TRIALS=1
TASK=wmt_14:language_pair=de-en
METRIC=bleu_4

#other configs
NUM_BEAMS_LIST="1 5"
MODELS="simple/model1"
MAX_EVAL_INSTANCES=10




cat ./test_run_all.ksh
echo -e "\n\n\n\n\n\n\n\n\n\n\n\n "
for MODEL in $MODELS; do
    for NUM_BEAMS in $NUM_BEAMS_LIST; do
        SUITE_NAME="${SUITE_BASE}_${NUM_BEAMS}"
        # RUN_ENTRY=wmt_14:language_pair=de-en,model=$MODEL,follow_format_instructions=instruct,num_beams=$NUM_BEAMS
        RUN_ENTRY=${TASK},model=${MODEL},follow_format_instructions=instruct,num_beams=$NUM_BEAMS 
        echo "RUN ENTRY IS $RUN_ENTRY"
        helm-run --run-entries $RUN_ENTRY --num-train-trials $NUM_TRAIN_TRIALS --max-eval-instances $MAX_EVAL_INSTANCES --suite $SUITE_NAME
        

        echo -e "\n\n\n\n\n\n\n\n\n\n\n\n <><><><><><><><><><><><><>"

        python process_results.py --model $MODEL --task  $TASK --num_beams $NUM_BEAMS  --metric $METRIC \
            --suite_name $SUITE_NAME --output_csv $OUTPUT_CSV
    done

done

# helm-summarize --suite $suite


echo helm-server --suite $suite
echo http://localhost:8000/ 
echo "\n\n\n\n"
# helm-summarize --suite beam_1739389179
#helm-server --suite beam_1739389179



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
