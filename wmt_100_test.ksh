#based on this page
#https://crfm-helm.readthedocs.io/en/latest/reproducing_leaderboards/


#ok, from my understanding this actually runs the test with simple/model1
#is it worth doing it for all 6 trials?

# Pick any suite name of your choice
export SUITE_NAME="wmt_cs_en_test_100"

# Replace this with your model or models

export RUN_ENTRIES_CONF_PATH=run_setup/wmt.conf
export SCHEMA_PATH=run_setup/schema_lite.yaml
export NUM_TRAIN_TRIALS=1

#test
#export MODELS_TO_RUN=stas/tiny-random-llama-2
#export MAX_EVAL_INSTANCES=10
#export PRIORITY=1

#real
export MODELS_TO_RUN=meta-llama/Llama-3.1-8B 
export MAX_EVAL_INSTANCES=1000
export PRIORITY=2

# echo helm-run --conf-paths $RUN_ENTRIES_CONF_PATH --num-train-trials $NUM_TRAIN_TRIALS --max-eval-instances $MAX_EVAL_INSTANCES --priority $PRIORITY --suite $SUITE_NAME --models-to-run $MODELS_TO_RUN


# helm-run --conf-paths $RUN_ENTRIES_CONF_PATH --num-train-trials $NUM_TRAIN_TRIALS --max-eval-instances $MAX_EVAL_INSTANCES --priority $PRIORITY --suite $SUITE_NAME --models-to-run $MODELS_TO_RUN
helm-run --run-entries wmt_14:language_pair=cs-en,model=$MODELS_TO_RUN,follow_format_instructions=instruct,num_beams=100 --num-train-trials $NUM_TRAIN_TRIALS --max-eval-instances $MAX_EVAL_INSTANCES --suite $SUITE_NAME

# helm-summarize --schema $SCHEMA_PATH --suite $SUITE_NAME
# echo helm-server --suite $SUITE_NAME
# helm-server --suite $SUITE_NAME
