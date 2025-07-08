RUN_PATH=./snellius_copies/helm_output/fairseq/wmt_14_language_pair_de_en_/fairseq_softmax/1_beams/runs/fairseq
SNELLIUS_METRIC="example_comet"
python append_snellius_metrics.py  --run_path ${RUN_PATH} --metric_name $SNELLIUS_METRIC