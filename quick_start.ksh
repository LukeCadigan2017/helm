# Run benchmark
#!/bin/bash



timestamp=$(date +%s)
SUITE=quickstart
MODEL=meta-llama/Llama-3.1-8B

NUM_BEAMS=1
NUM_TRAIN_TRIALS=1
MAX_EVAL_INSTANCES=5

rm completions.txt
helm-run --run-entries wmt_14:language_pair=cs-en,model=$MODEL,follow_format_instructions=instruct,num_beams=${NUM_BEAMS} --num-train-trials $NUM_TRAIN_TRIALS \
    --max-eval-instances $MAX_EVAL_INSTANCES --suite $SUITE --cache-instances
cat completions.txt

# echo helm-run --run-entries wmt_14:language_pair=cs-en,model=$MODEL,follow_format_instructions=instruct,num_beams=${NUM_BEAMS} --num-train-trials $NUM_TRAIN_TRIALS \
#     --max-eval-instances $MAX_EVAL_INSTANCES --suite $SUITE


#helm-summarize --suite $suite
#echo helm-server --suite $suite
