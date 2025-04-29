# from helm.benchmark.runner import InstanceGenerations,GenerationSummary
# from process_gens import get_gen_summary, get_run_folder, get_gen_summary_from_path

# from PostMetric import calculate_post_metric, get_post_metrics
# from helm.common.general import ensure_directory_exists, write, asdict_without_nones
# import os.path
# from comet import download_model, load_from_checkpoint
# from helm.common.gpu_utils import get_torch_device_name



import os
import sys
import json
import argparse
import logging
import math 

import sys
import json
import argparse


from tqdm import tqdm
# from stats import Correlation

PROMPT_W_ADD = "###Instruction###\n\
Please act as an impartial and helpful evaluator for natural language generation (NLG), and the audience is an expert in the field.\n\
Your task is to evaluate the quality of {task} strictly based on the given evaluation criterion.\n\
Begin the evaluation by providing your analysis concisely and accurately, and then on the next line, start with \"Rating:\" followed by your rating on a Likert scale from 1 to 5 (higher means better).\n\
You MUST keep to the strict boundaries of the evaluation criterion and focus solely on the issues and errors involved; otherwise, you will be penalized.\n\
Make sure you read and understand these instructions, as well as the following evaluation criterion and example content, carefully.\n\
\n\
###Evaluation Criterion###\n\
{aspect}\n\
\n\
###Example###\n\
{source_des}:\n\
{source}\n\
\n\
{addition_des}:\n\
{addition}\n\
\n\
{target_des}:\n\
{target}\n\
\n\
###Your Evaluation###\n"

PROMPT = "###Instruction###\n\
Please act as an impartial and helpful evaluator for natural language generation (NLG), and the audience is an expert in the field.\n\
Your task is to evaluate the quality of {task} strictly based on the given evaluation criterion.\n\
Begin the evaluation by providing your analysis concisely and accurately, and then on the next line, start with \"Rating:\" followed by your rating on a Likert scale from 1 to 5 (higher means better).\n\
You MUST keep to the strict boundaries of the evaluation criterion and focus solely on the issues and errors involved; otherwise, you will be penalized.\n\
Make sure you read and understand these instructions, as well as the following evaluation criterion and example content, carefully.\n\
\n\
###Evaluation Criterion###\n\
{aspect}\n\
\n\
###Example###\n\
{source_des}:\n\
{source}\n\
\n\
{target_des}:\n\
{target}\n\
\n\
###Your Evaluation###\n"

SEP = "<sep of eval.py>"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="[%(asctime)s,%(msecs)d] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s]   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

class Namespace(argparse.Namespace):
    model: str
    test_dir: str
    output_dir: str

    ## the config of vllm
    max_new_tokens: int
    temperature: float
    sampling_n: int
    tensor_parallel_size: int
    max_num_seqs: int

    ## whether output the correlation between human ratings and model evaluations
    correlation: bool

def generate(engine, sampling_params, test_prompts, use_tqdm: bool = True):
    
    for ex in test_prompts:
        test_prompt = ex[0]
        request_id = ex[1]
        engine.add_request(request_id, test_prompt, sampling_params)

    if use_tqdm:
        num_requests = engine.get_num_unfinished_requests()
        pbar = tqdm(total=num_requests, 
                    desc="Processed prompts", 
                    dynamic_ncols=True)
    
    outs = []
    while engine.has_unfinished_requests():
        step_outputs = engine.step()

        for step_output in step_outputs:
            if step_output.finished:
                outs.append(step_output)
                if use_tqdm:
                    pbar.update(1)
    
    if use_tqdm:
        pbar.close()

    return [([ex.text for ex in out.outputs], out.request_id) for out in outs]


def get_prompt(ex):
    if ex.get("addition", None) is not None:
        return PROMPT_W_ADD.format_map(ex)
    return PROMPT.format_map(ex)




def process(inputs):

    from vllm import EngineArgs, LLMEngine, SamplingParams
    max_num_batched_tokens=max(args.max_num_seqs, args.max_new_tokens)
    engine_args = EngineArgs(model=args.model, 
                              tensor_parallel_size=args.tensor_parallel_size,
                              max_num_seqs=args.max_num_seqs,
                              max_num_batched_tokens=max_num_batched_tokens,
                              max_model_len=max_num_batched_tokens,
                              gpu_memory_utilization=0.98,
                              swap_space=16)

    engine = LLMEngine.from_engine_args(engine_args)
    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, temperature=args.temperature, n=args.sampling_n)
    return generate(engine, sampling_params, inputs)




from typing import List
from dataclasses import dataclass, field



@dataclass(frozen=False)
class GeneratedOutput:
    """A `GeneratedOutput` is a single generated output that may contain text or multimodal content."""

    # The concatenation of all the tokens
    text: str


@dataclass(frozen=False)
class InstanceGenerations:
    """Split (e.g., train, valid, test)"""
    instance_id: str
    """id of instance"""

    prompt: str
    """Prompt used"""

    examples: List[GeneratedOutput]=None
    """List of unscored examples"""

    stats_dict: dict[str, any]=None
    
    evaluation: str = None


@dataclass(frozen=False)
class GenerationSummary:
    instance_generations :List[InstanceGenerations]

instance_generations=[]
for i in range(3):
  instance_generation=InstanceGenerations(
      instance_id=f"{i}", 
      prompt=f"Add 3 + {i}",
      examples = [
          GeneratedOutput(text=f"{3+i}"),
          GeneratedOutput(text=f"{3+i}. The answer is obvious.")
      ])
  instance_generations.append(instance_generation)
            

generation_summary = GenerationSummary(instance_generations=instance_generations)


args = Namespace(
    test_dir="",
    output_dir="",
    max_num_seqs=1024,
    max_new_tokens=2048,
    model="PKU-ONELab/Themis",
    temperature=0,
    sampling_n=1,
    tensor_parallel_size=4,
    correlation=False)    

all_test_prompts=[]

def get_example_id(instance_generation,output_num ):
    return f"{instance_generation.instance_id}{SEP}{output_num}"

for instance_generation in generation_summary.instance_generations:
    for output_num, generated_output in enumerate(instance_generation.examples):
        id=get_example_id(instance_generation,output_num )
        ex={
            "task": "Instruction Following",  # Which NLG task does the sample belongs to, e.g. Summarization
            "aspect": "Overall Quality",  # The criterion of the evaluation aspect, e.g. Fluency: Measure the quality of individual sentences of the summary...
            "source_des": "Instruction",  # The description of the source, e.g. Article
            "source": instance_generation.prompt.strip(),  # The source content
            "target_des": "Response",  # The description of the target, e.g. Summary
            "target": generated_output.text.strip(), # The target content
        }
        prompt=get_prompt(ex)
        all_test_prompts.append((prompt, id))


print(f"Instance generation: {generation_summary.instance_generations}")

print("Attempt to process")
outs = process(all_test_prompts)

id_to_eval = {}
for ex in outs:
    text, id = ex
    id_to_eval[id]=text

def parse(out: str):
    last_line = out.split('\n')[-1]
    if last_line.startswith("Rating: "):
        try: 
            rating = float(last_line[8:])
            if math.isfinite(rating):
                return {"Analysis": '\n'.join(out.split('\n')[:-1]), "Rating": rating}
        except:
            pass

    return {"Analysis": out, "Rating": 0}

for instance_generation in generation_summary.instance_generations:
    for output_num, generated_output in enumerate(instance_generation.examples):
        output_id=get_example_id(instance_generation,output_num )
        out=id_to_eval[output_id]
        parsed_dict=parse(out)
        generated_output.evaluation=out

        # {"Analysis": out, "Rating": 0}
        generated_output.stats_dict["example_themis"]= parsed_dict["Rating"]