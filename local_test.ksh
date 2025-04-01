#./test_run_all.ksh meta-llama/Llama-3.1-8B-Instruct 2 1
eval "$(conda shell.bash hook)"
conda activate crfm-helm2

# #model beam_num num_eval
# rm -rf eval_1/wmt/distilbert_distilgpt2/2_beams
./test_run_all.ksh wmt distilbert/distilgpt2 2 1
