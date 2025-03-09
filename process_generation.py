import json

import argparse

parser = argparse.ArgumentParser("simple_example")
parser.add_argument("--gen_output_json", help="model used for task", type=str)
args = parser.parse_args()

read_file=args.gen_output_json
write_file = args.gen_output_json.replace(".json",".txt")
print("Process generation: writing to ",write_file)
with open(read_file, "r") as read_f:
  with open(write_file, "a") as write_f:
    for line in read_f:
      sentence_details=json.loads(line)
      write_f.write(f"Prompt: {sentence_details['prompt']}\n")
      completions=sentence_details["completions"]
      for completion in completions:
        write_f.write(f"\tCompletion (Log_P={str(completion['s_logprob'])}): {completion['text']}\n")

      




#write lines


# python process_generation.py --gen_output_json benchmark_output/run_all_eval_10/generated_RUN_ENTRY.json --gen_output_txt benchmark_output/run_all_eval_10/generated_RUN_ENTRY.json
