# grammar:path=src/helm/benchmark/scenarios/best_chatgpt_prompts.yaml,tags=,model=instruction_following,num_respondents=1

#{description: "self_instruct:model=instruction_following,num_respondents=1", priority: 1}
#{description: "grammar:path=src/helm/benchmark/scenarios/best_chatgpt_prompts.yaml,tags=,model=instruction_following,num_respondents=1", priority: 1}
#{description: "open_assistant:language=en,model=instruction_following,num_respondents=1", priority: 1}
#{description: "vicuna:model=instruction_following,num_respondents=1", priority: 1}
#{description: "koala:model=instruction_following,num_respondents=1", priority: 1}
#{description: "anthropic_hh_rlhf:subset=hh,model=instruction_following,num_respondents=1", priority: 1}

helm-run --run-entries vicuna:model=distilbert/distilgpt2,num_respondents=1,num_beams=2 \
    --num-train-trials 2 --max-eval-instances 1 \
    -o helm_output/eval_1/instruct/distilbert_distilgpt2/2_beams \
    --suite eval_1 --disable-cache