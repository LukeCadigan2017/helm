# Run benchmark
#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate crfm-helm

timestamp=$(date +%s)
suite="beam_$timestamp"
model=simple/model1
eval_instances=10


echo model is $model
echo eval_instances is $eval_instances

#I think I searched either vicuna or rometheus to get this
helm-run --run-entries self_instruct:model=$model,num_respondents=1 --suite $suite  --max-eval-instances $eval_instances
