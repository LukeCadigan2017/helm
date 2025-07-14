import pandas as pd
from helm.benchmark.runner import InstanceGenerations,GenerationSummary
from typing import Any, List
import json
from helm.common.request import (GeneratedOutput, Token)

import PostMetric
import pandas as pd

from helm.benchmark.metrics.statistic import Stat
from typing import Dict, Optional

from helm.benchmark.augmentations.perturbation_description import (
    PerturbationDescription)
from dataclasses import dataclass
import os
import re


def print_keys(element, depth=0):

    # print(f"element: {type(element)}")
    
    if isinstance(element, dict) and element:
        first_key=next(iter(element.keys()))
        print(f"key {depth} example: {first_key}")
        print_keys(element[first_key], depth+1)
    else:
        print(f"value is {element}")


def fix_example_themis(completionExample):
    
    if(completionExample.stats_dict and "example_themis" in completionExample.stats_dict.keys()):
        def parse(evaluation):
         
            match = re.search(r'\brating\s*:\s*([1-5])\b', evaluation, re.IGNORECASE)

            if match:
                rating = int(match.group(1))
                return rating
            else:
                return -1
        rating = parse(completionExample.evaluation)
        rating = rating if rating is not None else -1
        completionExample.stats_dict["example_themis"]=rating
    return completionExample








def get_process_gen_params(test_name):

    def get_metrics(mode):
        compare_metric=None
        if(mode=="wmt"):
            task_names=["wmt_14_language_pair_de_en_"]
            custom_metrics=[]
            instance_metrics=["comet"]
            compare_metric="example_comet"

        elif(mode=="gsm"):
            task_names=["gsm_"]
            custom_metrics=[PostMetric.EXAMPLE_FINAL_NUM_EXACT_MATCH_METRIC(), PostMetric.EXAMPLE_EXACT_MATCH()]
            instance_metrics=[]
            compare_metric="final_num_exact_match"
            # instance_metrics=["exact_match_indicator","final_number_exact_match"]
        elif(mode=="instruct"):
            print("\n\n----------------\n NOTE: ONLY PRINTING 4 tasks ----------------\n")
            # task_names=["open_assistant:language=en,num_respondents=1,","self_instruct:num_respondents=1,"]
            task_names=[
                        "self_instruct_num_respondents_1_",
                        "anthropic_hh_rlhf_subset_hh_num_respondents_1_",
                        "vicuna_num_respondents_1_",
                         "koala_num_respondents_1_", 
                        "anthropic_hh_rlhf_subset_red_team_num_respondents_1_",
                        "grammar_path_src_helm_benchmark_scenarios_best_chatgpt_prompts.yaml_tags_num_respondents_1_"
                        ]
            custom_metrics=[]
            instance_metrics=[]
            compare_metric="example_themis"
        else:
            raise Exception(f"Did not recognize mode {mode}")
        assert isinstance(task_names, list)
        assert isinstance(task_names[0],str)
        return task_names, custom_metrics, instance_metrics, compare_metric

    root_folder=f"snellius_copies/helm_output"


    ####### all in one go 1000

    #0- 8b llama8
    #1 - 1B (olmo, llama)
    #2 - 2 olmo (7,13)
    #3 - special type

    override_task_names=None
    override_custom_metrics=None
    override_instance_metrics=None
    override_compare_metric=None


    if(test_name=="wmt_samples0"):
        mode = "wmt"
        num_beams_list=[1]
        suite_name="sample_100_eval_1000"
        models=["meta_llama_Llama_3.1_8B_Instruct"]

    ##### all in one go 500
    elif(test_name=="wmt_samples1"):
        mode = "wmt"
        suite_name="sample_100_eval_500_first_inst_0"
        num_beams_list=[1]
        models=["allenai_OLMo_2_0425_1B_Instruct","meta_llama_Llama_3.2_1B_Instruct"]

    elif(test_name=="wmt_samples2_1"):
        mode = "wmt"
        suite_name="sample_100_eval_100_first_inst_0"
        num_beams_list=[1]
        models=["allenai_OLMo_2_1124_7B_Instruct","allenai_OLMo_2_1124_13B_Instruct"]

    elif(test_name=="wmt_samples2_2"):
        mode = "wmt"
        suite_name="sample_100_eval_400_first_inst_100"
        num_beams_list=[1]
        models=["allenai_OLMo_2_1124_7B_Instruct","allenai_OLMo_2_1124_13B_Instruct"]

    elif(test_name=="wmt_samples3"):
        mode = "wmt"
        suite_name="sample_100_eval_500_first_inst_0"
        num_beams_list=[1]
        models=[
            "meta_llama_Llama_3.2_1B_Instruct",
        ]

    elif(test_name=="wmt_samples4"):
        mode = "wmt"
        suite_name="sample_100_eval_500_first_inst_0"
        num_beams_list=[1]
        models=[
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-1.7B",
            "Qwen/Qwen3-4B",
            "Qwen/Qwen3-8B",
            # "Qwen/Qwen3-32B",
        ]

    elif(test_name=="wmt_samples5"):
        mode = "wmt"
        suite_name="sample_100_eval_500_first_inst_0"
        num_beams_list=[1]
        models=[
            #dpo / sft
            "allenai_OLMo_2_1124_7B_SFT", 'allenai_OLMo_2_1124_7B_DPO',
            "allenai_OLMo_2_1124_13B_DPO", "allenai_OLMo_2_1124_13B_SFT",
            
            #base models
            "meta_llama_Llama_3.2_1B",
            "meta_llama_Llama_3.1_8B", 
            "allenai_OLMo_2_0425_1B",
            "allenai_OLMo_2_1124_7B",
            "allenai_OLMo_2_1124_13B",
        ]


    elif(test_name=="only_qwen_8b"):
        mode = "wmt"
        suite_name="sample_100_eval_500_first_inst_0"
        num_beams_list=[1]
        models=[
            "Qwen/Qwen3-8B"
        ]



    elif(test_name=="llama_template"):
        mode = "wmt"
        num_beams_list=[1]
        suite_name="llama_template"
        models=["meta_llama_Llama_3.1_8B_Instruct"]

    elif(test_name=="olmo_template"):
        mode = "wmt"
        num_beams_list=[1]
        suite_name="sample_100_eval_500_first_inst_0_template_true"
        models=["allenai_OLMo_2_1124_13B_Instruct"]


    elif(test_name=="gsm_samples1_1"):
        mode = "gsm"
        suite_name="sample_100_eval_20_first_inst_0"
        num_beams_list=[1]
        models=["meta_llama_Llama_3.1_8B_Instruct"]
    elif(test_name=="gsm_samples1_2"):
        mode = "gsm"
        suite_name="sample_100_eval_80_first_inst_20"
        num_beams_list=[1]
        models=["meta_llama_Llama_3.1_8B_Instruct"]
    elif(test_name=="gsm_samples1_3"):
        mode = "gsm"
        suite_name="sample_100_eval_400_first_inst_100"
        num_beams_list=[1]
        models=["meta_llama_Llama_3.1_8B_Instruct"]

    elif(test_name=="gsm_samples2"):
        mode = "gsm"
        suite_name="sample_100_eval_500_first_inst_0"
        num_beams_list=[1]
        models=["Qwen_Qwen3_8B"]

    elif(test_name=="fairseq_softmax"):
        mode = "wmt"
        suite_name="fairseq"
        num_beams_list=[1]
        models=["fairseq_softmax"]

    elif(test_name=="fairseq_sparsemax"):
        mode = "wmt"
        suite_name="fairseq"
        num_beams_list=[1]
        models=["fairseq_sparsemax"]

    
    elif(test_name=="instruct"):
        mode = "instruct"
        suite_name="sample_100_eval_100_first_inst_0"
        num_beams_list=[1]
        models=["Qwen_Qwen3_8B"]
        

    elif(test_name=="qwen_25_instruct"):
        mode = "instruct"
        suite_name="sample_100_eval_100_first_inst_0"
        num_beams_list=[1]
        models=["Qwen_Qwen2.5_7B_Instruct"]
        override_task_names=["koala_num_respondents_1_"]
        

    elif(test_name=="olmo_instruct"):
        mode = "instruct"
        suite_name="sample_100_eval_100_first_inst_0"
        num_beams_list=[1]
        models=["allenai_OLMo_2_1124_13B_Instruct"]

    elif(test_name=="current_test"):
        mode = "wmt"
        suite_name="sample_100_eval_500_first_inst_0"
        num_beams_list=[1]
        models=["meta-llama/Llama-3.1-8B"]



    else:
        except_str=f"Luke: task name {test_name} not found"
        print(except_str)
        raise Exception(except_str)
    
    task_names, custom_metrics, instance_metrics, compare_metric= get_metrics(mode)
    if override_task_names:
        task_names= override_task_names
    return root_folder, num_beams_list, models, custom_metrics, task_names, suite_name, instance_metrics, compare_metric
    

@dataclass(frozen=False)
class PerInstanceStats:
    """
    Captures a unit of evaluation.
    """

    # Uniquely identifies the input instance
    instance_id: str
    train_trial_index: int
    """Which replication"""

    stats: List[Stat]
    """Statistics computed from the predicted output"""
    perturbation: Optional[PerturbationDescription]=None


############ UTILS ############


def assert_dir_exists(dir_name):
    dirs=dir_name.split("/")
    for i in range(len(dirs)):
        prev_dir="/".join(dirs[:i])
        cur_dir="/".join(dirs[:i+1])
        if not os.path.isdir(cur_dir):
            error_str="\n------------------------------------------\n"
            error_str+="Error:\n"
            error_str+=f"dir_name does not exist: {dir_name}\n\n"
            error_str+=f"Directory exists: {prev_dir}\n\n"
            error_str+=f"Extension does not exist: {dir_name[len(prev_dir)+1:]}\n"
            error_str+=f"To check:\n"
            error_str+=f"ls {prev_dir}"
            raise Exception(error_str)
        


def clean_str_for_os(str_to_clean:str):
    str_to_clean=str_to_clean.strip()
    chars = ["=",",",":", "__", "-", "/"]
    for char in chars:
        str_to_clean=str_to_clean.replace(char,"_")
    return str_to_clean


def get_run_folder(root_folder:str, num_beams:int, model:str, task_name: str, suite_name:str):
    
    num_beams=clean_str_for_os(str(num_beams))
    model=clean_str_for_os(model)
    task_name=clean_str_for_os(task_name)
    suite_name=clean_str_for_os(suite_name)

    run_folder= f"{root_folder}/{suite_name}/{task_name}/{model}/{num_beams}_beams/runs/{suite_name}"
    assert_dir_exists(run_folder)
    return run_folder



############ Gen Summary stuff ############

def get_completion_from_examples(examples):
    examples.sort(key=lambda x:float(x.logprob),reverse=True)
    completion=examples[0].text
    completion_logprob=examples[0].logprob
    return examples, completion, completion_logprob

def clean_generation_summary(generationSummary:GenerationSummary)->GenerationSummary:
    def clean_instance_generation(instanceGenerations:InstanceGenerations)->InstanceGenerations:
        def clean_generated_output(generatedOutput:GeneratedOutput)-> GeneratedOutput:
            generatedOutput.text=truncate_sequence(generatedOutput.text)
            generatedOutput=fix_example_themis(generatedOutput)
            return generatedOutput
        # print(f"examples len is {len(instanceGenerations.examples)}")
        instanceGenerations.examples=[clean_generated_output(generatedOutput=example) for example in instanceGenerations.examples]
        instanceGenerations.examples.sort(key=lambda x:float(x.logprob),reverse=True)
        completion=instanceGenerations.examples[0]
        instanceGenerations.completion=completion.text
        instanceGenerations.completion_logprob=completion.logprob
        return instanceGenerations
    generationSummary.instance_generations=[clean_instance_generation(instanceGenerations=instance_generation) for instance_generation in generationSummary.instance_generations]
    # assert len(generationSummary.instance_generations)==eval_instances
    # print(f"number of instances: {len(generationSummary.instance_generations)}")
    return generationSummary



def get_gen_summary_from_path(path) -> GenerationSummary:
    # print(f"path is {path}")
    def json_to_instance_generation(instance_dict:dict) -> InstanceGenerations:
        def json_to_generated_output(generated_output_dict):
            generated_output=GeneratedOutput(**generated_output_dict)
            tokens = [Token(**token) for token in generated_output.tokens]
            generated_output.tokens=tokens
            return generated_output
        instance_generation = InstanceGenerations(**instance_dict)
        examples = [ json_to_generated_output(generated_output_dict) for generated_output_dict in instance_generation.examples]
        instance_generation.examples=examples
        return instance_generation
    # print(f"Getting gen summary from: {path}")
    with open(path,'r') as json_file:
        generation_summary_dict=json.load(json_file)
    generation_summary=GenerationSummary(**generation_summary_dict)
    instance_generations = [json_to_instance_generation(instance_dict)  for instance_dict in generation_summary.instance_generations ]
    generation_summary.instance_generations=instance_generations

    generation_summary=clean_generation_summary(generation_summary)
    return generation_summary


def get_gen_summary_from_run_folder(run_folder: str):
    gen_sum_raw_path=f"{run_folder}/generation_summary.json"
    gen_sum_metric_path=f"{run_folder}/generation_summary_metrics.json"
    input_path = gen_sum_metric_path if os.path.isfile(gen_sum_metric_path) else gen_sum_raw_path
    generation_summary=get_gen_summary_from_path(input_path)
    return generation_summary


def truncate_sequence(text:str, all_stops=["<|end_of_text|>"]) -> str:
    for stop in all_stops:
        try:
            text = text[: text.index(stop)]
        except ValueError:
            pass
    return text.strip()



def append_to_dict(dict, key_list, value):
    cur_key=key_list[0]
    
    #make sure it exists
    if cur_key not in dict.keys():
        dict[cur_key]={}

    #append recursively if not
    if(len(key_list)>1):
        append_to_dict(dict[cur_key], key_list[1:], value)
    else:
        dict[cur_key]=value

def calculate_dict(init_dict, root_folder, num_beams_list:List[int], models:List[float], task_names:List[str], suite_name:str, dict_function, print_files=False)->Dict[int, GenerationSummary]:
    for model in models:        
        for task_name in task_names:
            for num_beams in num_beams_list:
                run_folder=get_run_folder(root_folder=root_folder, num_beams=num_beams, model=model, task_name=task_name, suite_name=suite_name)
                if(print_files):
                    print(run_folder)
                obj=dict_function(run_folder)
                append_to_dict(init_dict, [suite_name, model, task_name, num_beams], obj)
    return init_dict


def calculate_instance_stats_dict(init_dict, root_folder, num_beams_list:List[int], models:List[float], task_names:List[str], suite_name:str, instance_metrics:List[str])->Dict[int, List[PerInstanceStats]]:
    def json_to_run_instance_stats(run_folder, instance_metrics) -> List[PerInstanceStats]:
        path=run_folder+"/per_instance_stats.json"
        # print(f"Analyzing path: {path}")
        if not os.path.isfile(path):
            return None
        with open(path,'r') as json_file:
            list_instance_stats_dicts=json.load(json_file)
        
        instance_id_to_stats_dict={}
        for list_instance_stats_dict in list_instance_stats_dicts:
            per_instance_stats = PerInstanceStats(**list_instance_stats_dict)
            stats = [Stat(**stat_dict) for stat_dict in per_instance_stats.stats]
            per_instance_stats.stats=stats
            stats_dict={}
            for stat in per_instance_stats.stats:
                name = stat.name
                if name["name"] in instance_metrics and name["split"]=="test" and "perturbation" not in name.keys():
                    stats_dict[name["name"]]= stat.mean
            instance_id_to_stats_dict[per_instance_stats.instance_id]=stats_dict
        return instance_id_to_stats_dict
    dict_function = lambda run_folder: json_to_run_instance_stats(run_folder=run_folder, instance_metrics=instance_metrics)
    return calculate_dict(init_dict, root_folder, num_beams_list, models, task_names, suite_name, dict_function)

def calculate_instances_dict(init_dict, root_folder, num_beams_list:List[int], models:List[float], task_names:List[str], suite_name:str, print_files:bool):
    def get_instance_dict_from_run_folder(run_folder):
        gen_summary= get_gen_summary_from_run_folder(run_folder)
        instance_dict={}
        for instance_generation in gen_summary.instance_generations:
            instance_dict[instance_generation.instance_id] = instance_generation
        return instance_dict
    print_files=True
    return calculate_dict(init_dict,root_folder, num_beams_list, models, task_names, suite_name,get_instance_dict_from_run_folder, print_files)

get_first = lambda x: next(iter(x.values()))
# @classmethod  
# def get_instance_info(self, root_folder, num_beams_list:List[int], models:List[str], task_name: str, suite_name:str)->Dict[int, GenerationSummary]:
#     num_beams=num_beams_list[0]
#     model=models[0]
#     instance_infos= {}
#     instance_metrics=[PostMetric.ReferenceMetric()]

#     generation_summary=get_gen_summary(root_folder=root_folder, num_beams=num_beams, model=model, task_name=task_name, suite_name=suite_name)
#     for instance_generation in generation_summary.instance_generations:
#         instance_id=instance_generation.instance_id
#         if instance_id not in instance_infos.keys():
#             instance_dict={}
#             for metric in instance_metrics:
#                 instance_dict=PostMetric.calculate_post_metric(metrics_dict=instance_dict,metric=metric,instance_generation=instance_generation,generated_output=None)
#             instance_infos[instance_id]=instance_dict
#     return instance_infos


# @classmethod  
# def get_metrics_df(self, root_folder):

#     try:
#         metrics_file=f"{root_folder}/metrics_csv.txt"
#         raw_metric_df = pd.read_csv(metrics_file, header=None)
#         raw_metric_df.columns=[ "model", "task", "beam_num", "metric", "value"]
#         raw_metric_df.drop(["task"],axis=1)
#         metric_df = raw_metric_df.pivot(
#             index=["model", "beam_num"],
#             columns="metric",
#             values="value"
#         ).reset_index()
#         metric_df.sort_values("beam_num")
#         self.metric_df=metric_df
#         return metric_df
#     except:
#         return None



class ProcessGens:
    root_folder:str
    # task_and_beam_num_to_summary:Dict[int, GenerationSummary]
    instances_dict={}
    instance_stats_dict={}
    prompt_to_instanceID={}
    metrics_dict:List[Dict[str,any]]=[]
    first_run_instances={}


    def __init__(self):
        self.instances_dict={}
        self.instance_stats_dict={}
        self.prompt_to_instanceID={}
        pass

    def init_with_mode(self, process_gens_modes:List[str], print_files=False):
        print(f"Init: process_gens_mode {process_gens_modes}")
        if isinstance(process_gens_modes, str):
            process_gens_modes = [process_gens_modes]
        for process_gens_mode in process_gens_modes:
            root_folder, num_beams_list, models, custom_metrics, task_names, suite_name, instance_metrics, compare_metric= get_process_gen_params(process_gens_mode)
            self.init(root_folder=root_folder,num_beams_list=num_beams_list,models=models,custom_metrics=custom_metrics,task_names=task_names,  suite_name=suite_name,instance_metrics=instance_metrics, print_files=print_files, compare_metric=compare_metric)
            
    def get_params(self):
        root_folder     =self.process_gen_params["root_folder"]
        num_beams_list  =self.process_gen_params["num_beams_list"]
        models          =self.process_gen_params["models"]
        custom_metrics  =self.process_gen_params["custom_metrics"]
        task_names      =self.process_gen_params["task_names"]
        suite_name      =self.process_gen_params["suite_name"]
        instance_metrics=self.process_gen_params["instance_metrics"]

        compare_metric=self.process_gen_params["compare_metric"]
        return root_folder, num_beams_list, models, custom_metrics, task_names, suite_name, instance_metrics, compare_metric

    def calculate_instance_id(self, instance_generation):
        key=instance_generation.prompt
        if key not in self.prompt_to_instanceID:
            self.prompt_to_instanceID[key]= len(self.prompt_to_instanceID)
        return self.prompt_to_instanceID[key]

            

    def get_metrics_dict(self, instances_dict:Dict[int, GenerationSummary], custom_metrics:List[PostMetric.PostMetric], instance_stats_dict):

        base_metrics=[PostMetric.TextMetric,PostMetric.SentenceLenMetric(),PostMetric.OutputProbMetric(),PostMetric.IsCompletionMetric()]
        metrics=base_metrics+custom_metrics
        metrics_dicts=[]   

        for suite_name in instances_dict.keys():
            for model in instances_dict[suite_name].keys():        
                for task_name in instances_dict[suite_name][model].keys():
                    for beam_num in instances_dict[suite_name][model][task_name].keys():

                        instance_stats_per_run = instance_stats_dict[suite_name][model][task_name][beam_num]

                        for instance_id, instance_generation in instances_dict[suite_name][model][task_name][beam_num].items():
                            for example_idx,generated_output in enumerate(instance_generation.examples):
                                pd_metrics_dict=generated_output.stats_dict if generated_output.stats_dict  is not None else {} 
                                
                                pd_metrics_dict["beam_num"]=beam_num
                                pd_metrics_dict["task_name"]=task_name
                                pd_metrics_dict["model"]=model
                                pd_metrics_dict["example_idx"]=example_idx
                                pd_metrics_dict["rank"]=100-example_idx
                                pd_metrics_dict["suite"]=suite_name
                                
                                pd_metrics_dict["instanceID"]=self.calculate_instance_id(instance_generation)

                                #fill out the metrics dict
                                for metric in metrics:
                                    pd_metrics_dict=PostMetric.calculate_post_metric(pd_metrics_dict,metric,instance_generation,generated_output)
                                
                                if(example_idx==0):
                                    pd_metrics_dict["isCompletion"]=(example_idx==0)
                                    if(instance_stats_per_run and instance_generation.instance_id in instance_stats_per_run.keys()):
                                        completion_metrics_dict = instance_stats_per_run[instance_generation.instance_id]
                                        for stat_name, value in completion_metrics_dict.items():
                                            pd_metrics_dict[stat_name]= value
                                metrics_dicts.append(pd_metrics_dict)
        return metrics_dicts

    def init(self,root_folder:str, num_beams_list:List[int], models:List[float], custom_metrics:List[PostMetric.PostMetric],task_names:List[str], instance_metrics:Dict[int, Dict[str, Dict[str, float]]]=None, suite_name:str="", print_files:bool=False, compare_metric:str=""):
        
        # #this is the pre-computed metrics
        # print("get_metrics_df")
        # self.metrics_df=self.get_metrics_df(root_folder)
        # print("get_instance_info")

        # #this is th
        # instance_info=self.get_instance_info(root_folder=root_folder, num_beams_list=num_beams_list, models=models,task_name= task_name,suite_name=suite_name)
        # self.instance_info=instance_info

        #get the generation summary for each task beam
        print("calculate_gen_summary_dict")
        
        
        self.instances_dict=calculate_instances_dict(init_dict=self.instances_dict, root_folder=root_folder, num_beams_list=num_beams_list, models=models,task_names=task_names, suite_name=suite_name, print_files=print_files)
        

        #get the run instance stats for each task and beam
        self.instance_stats_dict=calculate_instance_stats_dict(init_dict=self.instance_stats_dict, root_folder=root_folder, num_beams_list=num_beams_list, models=models, task_names=task_names, suite_name=suite_name, instance_metrics=instance_metrics)
        

        print("get_metrics_dict")
        self.metrics_dicts=self.get_metrics_dict(instances_dict=self.instances_dict, custom_metrics=custom_metrics, instance_stats_dict=self.instance_stats_dict)


        self.first_run_instances=get_first(get_first(get_first(self.instances_dict)))
        self.ids= list(self.first_run_instances.keys())


        self.process_gen_params = {"root_folder":root_folder,
            "num_beams_list":num_beams_list,
            "models":models,
            "custom_metrics":custom_metrics,
            "task_names":task_names,
            "suite_name":suite_name,
            "instance_metrics":instance_metrics,
            "compare_metric":compare_metric
        }
        



    # elif(test_name=="wmt_samples_original"):
    #     mode = "wmt"
    #     suite_name="sample_10_eval_1000"
    #     num_beams_list=[1]
    #     models=["allenai_OLMo_2_0425_1B_Instruct","allenai_OLMo_2_1124_7B_Instruct","allenai_OLMo_2_1124_13B_Instruct","meta_llama_Llama_3.2_1B_Instruct","meta_llama_Llama_3.1_8B_Instruct"]

    # elif(test_name=="wmt_single"):
    #     mode = "wmt"
    #     suite_name="sample_100_eval_1000"
    #     num_beams_list=[1]
    #     models=["meta_llama_Llama_3.1_8B_Instruct"]

    # elif(test_name=="wmt_single_10"):
    #     mode = "wmt"
    #     suite_name="sample_10_eval_1000"
    #     num_beams_list=[1]
    #     models=["meta_llama_Llama_3.1_8B_Instruct"]

    # elif(test_name=="wmt_test"):
    #     mode = "wmt"
    #     suite_name="sample_10_eval_1000"
    #     num_beams_list=[1]
    #     models=["meta_llama_Llama_3.1_8B_Instruct"]

    # elif(test_name=="wmt_single_top_k_2"):
    #     mode = "wmt"
    #     suite_name="sample_10_eval_20_top_k_2"
    #     num_beams_list=[1]
    #     models=["meta_llama_Llama_3.1_8B_Instruct"]

    # elif(test_name=="wmt_top_k"):
    #     mode = "wmt"
    #     suite_name="sample_100_eval_100_top_k_30"
    #     num_beams_list=[1]
    #     models=["meta_llama_Llama_3.1_8B_Instruct"]

    # elif(test_name=="wmt_sample_50"):

    #     root_folder="snellius_copies/helm_output/notable_samples"
    #     mode = "wmt"
    #     suite_name="sample_return_100_eval_100"
    #     num_beams_list=[1]
    #     models=["meta_llama_Llama_3.1_8B_Instruct"]

    # elif(test_name=="wmt_sample_100"):

    #     root_folder="snellius_copies/helm_output/notable_samples"
    #     mode = "wmt"
    #     suite_name="sample_return_100_eval_100"
    #     num_beams_list=[1]
    #     models=["meta_llama_Llama_3.1_8B_Instruct"]

    # elif(test_name=="wmt_beam8"):
    #     mode = "wmt"
    #     suite_name="sample_1_eval_1000"
    #     num_beams_list=[8]
    #     models=["meta_llama_Llama_3.1_8B_Instruct", "allenai_OLMo_2_1124_13B_Instruct"]

    
    # elif(test_name=="wmt_beam128"):
    #     mode = "wmt"
    #     suite_name="sample_1_eval_1000"
    #     num_beams_list=[128]
    #     models=["meta_llama_Llama_3.1_8B_Instruct"]


    # elif (test_name=="full_instruct"):
    #     mode="instruct"
    #     suite_name="full_instruct_1_samples_100_evals"
    #     num_beams_list=[2,4,8]
    #     models=["allenai_OLMo_2_1124_13B_Instruct"]

    # elif (test_name=="instruct8"):
    #     mode="instruct"
    #     suite_name="full_instruct_1_samples_100_evals"
    #     num_beams_list=[8]
    #     models=["allenai_OLMo_2_1124_13B_Instruct"]

    # ###### INDIVIDUAL TESTS  ######
    # elif(test_name=="llama_gsm_sample"):
    #     mode = "gsm"
    #     suite_name="sample_10_eval_1000"
    #     num_beams_list=[1]
    #     models=["meta_llama_Llama_3.1_8B_Instruct"]
        
    # elif(test_name=="olmo_wmt"):
    #     mode = "wmt"
    #     suite_name="full_wmt_1_samples_1000_evals"
    #     num_beams_list=[2,4,8,16]
    #     models=["allenai_OLMo_2_1124_13B_Instruct"]
        
    
    # elif(test_name=="olmo_gsm"):
    #     mode = "gsm"
    #     suite_name="full_wmt_1_samples_1000_evals"
    #     num_beams_list=[2,4,8]
    #     models=["allenai_OLMo_2_1124_13B_Instruct"]

        
