conda activate crfm-helm2

source hf.env
huggingface-cli login --token $HF_TOKEN --add-to-git-credential


echo huggingface-cli whoami
huggingface-cli whoami

python append_snellius_metrics.py --num_beams 1 --model distilbert/distilgpt2 --eval_instances 4 \
    --task_name wmt_14:language_pair=de-en, --run_path helm_output/sample_return_4/wmt_14_language_pair_de_en_/distilbert_distilgpt2/1_beams/runs/sample_return_4 \
    --metric_name example_themis