#./test_run_all.ksh meta-llama/Llama-3.1-8B-Instruct 2 1
eval "$(conda shell.bash hook)"
conda activate crfm-helm2

#model beam_num num_eval
./test_run_all.ksh distilbert/distilgpt2 2 1
