import sys
import json
import argparse

parser = argparse.ArgumentParser("simple_example")
parser.add_argument("--model", help="model used for task", type=str)
parser.add_argument("--task", help="task name", type=str)
parser.add_argument("--num_beams", help="number of beams", type=str)
parser.add_argument("--metric", help="metric in question", type=str)

parser.add_argument("--suite_name", help="suite name", type=str)
parser.add_argument("--output_csv", help="suite name", type=str)
args = parser.parse_args()
# Usage example
# python process_results.py --model meta-llama_Llama-3.1-8B --task_name  wmt_14:language_pair=cs-en --num_beams 100  --metric bleu_4 \
#  --suite_name real_wmt_cs_en_test_100 --full_task wmt_14:language_pair=cs-en,model=meta-llama_Llama-3.1-8B \
#  --output_csv test_process.txt



model_clean=args.model.replace("/","_")

json_file=f"./benchmark_output/runs/{args.suite_name}/{args.task},model={model_clean}/stats.json"
print(f"Process results from json_file {json_file}. Saving to {args.output_csv}")

with open(json_file) as f:
    infos = json.load(f)

#parse json file
values=[]
for info in infos:
    name = info["name"]
    if name["name"]==args.metric and name["split"]=="test" and "perturbation" not in name.keys():
        values.append(info["mean"])

#save to file

split_char=","
eol_char="\n"

if(len(values)==1):
    value=values[0]
    data_vals=[args.model, args.task, args.num_beams, args.metric, str(value)]
    for data in data_vals:
        if(split_char in data or eol_char in data):
            raise Exception(f"Could not process data. Data contains splits_char.\n Data: {data} split_char: {split_char} or {eol_char}")

    save_str=split_char.join(data_vals)+eol_char
    with open(args.output_csv, "a") as text_file:
        print(f"Saving results: {save_str} to file {args.output_csv}")
        text_file.write(save_str)
    
else:
    print(f"Wrong number values return ({len(values)}). Could not process results for {[args.model, args.task, args.num_beams, args.metric]}")
    # raise Exception("Number of values is ",len(values))

