git pull origin main

source ~/miniconda3/bin/activate
conda activate crfm-helm2 

source hf.env
huggingface-cli login --token $HF_TOKEN --add-to-git-credential


echo huggingface-cli whoami
huggingface-cli whoami