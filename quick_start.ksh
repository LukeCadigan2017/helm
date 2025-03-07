# Run benchmark
#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate crfm-helm

timestamp=$(date +%s)
SUITE=quickstart
MODEL=stas/tiny-random-llama-2
# model="meta-llama/Llama-3.1-8B"
# model=openai/gpt2
EVAL_INSTANCES=3


# echo model is $model
# echo eval_instances is $eval_instances



# echo helm-run --run-entries wmt_14:language_pair=cs-en,model=$model,output_format_instructions=wmt_14 --suite $suite --max-eval-instances $eval_instances
helm-run --run-entries wmt_14:language_pair=de-en,model=$MODEL,follow_format_instructions=instruct,num_beams=$NUM_BEAMS --suite $SUITE  --max-eval-instances $EVAL_INSTANCES 

#helm-summarize --suite $suite
#echo helm-server --suite $suite
#echo "benchmark_output/runs/$suite/wmt_14\:language_pair\=cs-en\,model\=simple_model1/stats.json"
