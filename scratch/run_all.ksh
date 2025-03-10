#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate crfm-helm
#simple/model1

timestamp=$(date +%s)
#echo $timestamp

#num_beams="1 5 100"
#models="stas/tiny-random-llama-2 simple/model1"
#tasks="gsm_iom gsm_iom2"

num_beams="1"
models="simple/model1"
tasks="gsm gsm_iom"



suite="beam_$timestamp"



# helm-run --run-entries gsm_iom:model=stas/tiny-random-llama-2,num_beams=2  --suite my-suite --max-eval-instances 2

echo "NOTE: ONLY SOME EVAL INSTANCES"

for model in $models; do
    for task in $tasks; do
        for num_beam in $num_beams; do
            echo $model $num_beam $task
            echo helm-run --run-entries $task:model=$model,num_beams=$num_beam  --suite $suite --max-eval-instances 2
            #helm-run --run-entries $task:model=$model,num_beams=$num_beam  --suite $suite --max-eval-instances 2
            
            
            
            helm-run --run-entries $task:model=$model,num_beams=$num_beam  --suite $suite
            # echo "\n\n\n\n\n\n\n\n\n\n"
        done
    done
done

helm-summarize --suite $suite


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
