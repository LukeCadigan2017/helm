# Run benchmark
#!/bin/bash



timestamp=$(date +%s)
SUITE=quickstart
# MODEL=meta-llama/Llama-3.1-8B
MODEL=stas/tiny-random-llama-2

NUM_BEAMS=5
NUM_TRAIN_TRIALS=1
MAX_EVAL_INSTANCES=1

rm -f c.txt
helm-run --run-entries wmt_14:language_pair=de-en,model=$MODEL,follow_format_instructions=instruct,num_beams=${NUM_BEAMS},generated_output_file=c.txt --num-train-trials $NUM_TRAIN_TRIALS \
    --max-eval-instances $MAX_EVAL_INSTANCES --suite $SUITE --disable-cache
echo cat c.txt
cat c.txt


# echo helm-run --run-entries wmt_14:language_pair=cs-en,model=$MODEL,follow_format_instructions=instruct,num_beams=${NUM_BEAMS} --num-train-trials $NUM_TRAIN_TRIALS \
#     --max-eval-instances $MAX_EVAL_INSTANCES --suite $SUITE


#helm-summarize --suite $suite
#echo helm-server --suite $suite
