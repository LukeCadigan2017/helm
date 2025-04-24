import sys
import json
import argparse

parser = argparse.ArgumentParser("simple_example")
from helm.benchmark.runner import InstanceGenerations,GenerationSummary
from process_gens import get_gen_summary, get_run_folder, get_gen_summary_from_path

from PostMetric import calculate_post_metric, get_post_metrics
from helm.common.general import ensure_directory_exists, write, asdict_without_nones
import os.path
#input: helm_output/eval_1/wmt_14_language_pair_de_en_/distilbert_distilgpt2/1_beams
#       helm_output/eval_1/wmt_14_language_pair_de_en_/distilbert_distilgpt2/1_beams
#                  eval_1/wmt/distilbert/distilgpt2/1_beams/runs/eval_1/generation_summary.json



def get_metrics_from_names(metric_names):
    if(metric_names=="no_metrics"):
        return []
    metric_names_list=metric_names.split(",")
    return get_post_metrics(metric_names_list)

def fill_stats_dict(dict_wrapper, metrics,instance_generation,generated_output):

    stats_dict= {} if dict_wrapper.stats_dict is None else dict_wrapper.stats_dict
    for metric in metrics:
        stats_dict=calculate_post_metric(stats_dict,metric,instance_generation,generated_output)
    dict_wrapper.stats_dict=stats_dict
    return dict_wrapper



#parser
parser.add_argument("--model", help="model used for task", type=str)
parser.add_argument("--eval_instances", help="number of instances to eval", type=str)
parser.add_argument("--task_name", help="task name", type=str)
parser.add_argument("--num_beams", help="number of beams", type=str)
parser.add_argument("--instance_metrics", help="metrics in question", type=str)
parser.add_argument("--example_metrics", help="metrics in question", type=str)
parser.add_argument("--run_path", help="folder before eval_{beam_num}", type=str)
args = parser.parse_args()

#process args
num_beams=args.num_beams
model=args.model
task_name=args.task_name
eval_instances=args.eval_instances
run_path = args.run_path



instance_metrics=get_metrics_from_names(args.instance_metrics)
example_metrics=get_metrics_from_names(args.example_metrics)


#set up
gen_sum_raw_path=f"{run_path}/generation_summary.json"
gen_sum_metric_path=f"{run_path}/generation_summary_metrics.json"
input_path = gen_sum_metric_path if os.path.isfile(gen_sum_metric_path) else gen_sum_raw_path
generation_summary=get_gen_summary_from_path(input_path)

for instance_generation in generation_summary.instance_generations:
    instance_generation=fill_stats_dict(dict_wrapper=instance_generation, metrics=instance_metrics, instance_generation=instance_generation,generated_output=None)
    for idx,generated_output in enumerate(instance_generation.examples):
        generated_output=fill_stats_dict(dict_wrapper=generated_output, metrics=example_metrics,instance_generation=instance_generation,generated_output=generated_output)

write(
    gen_sum_metric_path,
    json.dumps(asdict_without_nones(generation_summary),indent=2)
)



#we have: helm_output/sample_return_2/wmt_14_language_pair_de_en_/distilbert_distilgpt2/1_beams/generation_summary.json
#we want: helm_output/sample_return_2/wmt_14_language_pair_de_en_/distilbert_distilgpt2/1_beams/runs/sample_return_2/generation_summary.json

#we want:   helm_output/eval_1/wmt_14_language_pair_de_en_/distilbert_distilgpt2/1_beams/runs/eval_1/generation_summary.json
#how to call:
# python append_snellius_metrics.py --num_beams 32 --model "meta_llama_Llama_3.1_8B_Instruct" --eval_instances 5 --task_name "wmt_14_language_pair_de_en_" \
#     --run_path "snellius_copies/wmt_test" --metric_names "test"

#examples:
# num_beams=32
# model="meta_llama_Llama_3.1_8B_Instruct"
# eval_instances=5
# task_name="wmt_14_language_pair_de_en_"
# run_path=f"snellius_copies/wmt_test"
# metric_names="test"
