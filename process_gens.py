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
        if(mode=="wmt"):
            task_names=["wmt_14_language_pair_de_en_"]
            custom_metrics=[ PostMetric.BLEU1_METRIC(),PostMetric.BLEU4_METRIC()]
            instance_metrics=["comet"]
        elif(mode=="gsm"):
            task_names=["gsm_"]
            custom_metrics=[PostMetric.EXAMPLE_FINAL_NUM_EXACT_MATCH_METRIC()]
            instance_metrics=[]
            # instance_metrics=["exact_match_indicator","final_number_exact_match"]
        elif(mode=="instruct"):
            print("\n\n----------------\n NOTE: ONLY PRINTING 4 tasks ----------------\n")
            # task_names=["open_assistant:language=en,num_respondents=1,","self_instruct:num_respondents=1,"]
            task_names=[
                        "anthropic_hh_rlhf_subset_hh_num_respondents_1_",
                         "koala_num_respondents_1_", 
                        "anthropic_hh_rlhf_subset_red_team_num_respondents_1_",
                        "self_instruct_num_respondents_1_",
                        "grammar_path_src_helm_benchmark_scenarios_best_chatgpt_prompts.yaml_tags_num_respondents_1_",
                        "vicuna_num_respondents_1_"]
            custom_metrics=[]
            instance_metrics=[]
        else:
            raise Exception(f"Did not recognize mode {mode}")
        assert isinstance(task_names, list)
        assert isinstance(task_names[0],str)
        return task_names, custom_metrics, instance_metrics

    root_folder=f"snellius_copies/helm_output"
    if(test_name=="wmt_samples"):
        mode = "wmt"
        # suite_name="sample_return_20_eval_500"
        # suite_name="sample_return_100_eval_100"
        suite_name="sample_10_eval_1000"
        num_beams_list=[1]
        # models=["meta_llama_Llama_3.1_8B_Instruct"]
        models=["allenai_OLMo_2_0425_1B_Instruct","allenai_OLMo_2_1124_7B_Instruct","allenai_OLMo_2_1124_13B_Instruct","meta_llama_Llama_3.2_1B_Instruct","meta_llama_Llama_3.1_8B_Instruct"]

    elif(test_name=="wmt_beam8"):
        mode = "wmt"
        suite_name="sample_1_eval_1000"
        num_beams_list=[8]
        models=["meta_llama_Llama_3.1_8B_Instruct", "allenai_OLMo_2_1124_13B_Instruct"]

    
    elif(test_name=="wmt_beam128"):
        mode = "wmt"
        suite_name="sample_1_eval_1000"
        num_beams_list=[128]
        models=["meta_llama_Llama_3.1_8B_Instruct"]

    # elif(test_name=="wmt_beam8_new"):
    #     mode = "wmt"
    #     suite_name="full_wmt_1_samples_1000_evals"
    #     num_beams_list=[16]
    #     models=["meta_llama_Llama_3.1_8B_Instruct"]

    elif (test_name=="full_instruct"):
        mode="instruct"
        suite_name="full_instruct_1_samples_100_evals"
        num_beams_list=[2,4,8]
        models=["allenai_OLMo_2_1124_13B_Instruct"]

    elif (test_name=="instruct8"):
        mode="instruct"
        suite_name="full_instruct_1_samples_100_evals"
        num_beams_list=[8]
        models=["allenai_OLMo_2_1124_13B_Instruct"]
        



    ###### INDIVIDUAL TESTS  ######
    elif(test_name=="llama_gsm_sample"):
        mode = "gsm"
        suite_name="sample_10_eval_1000"
        num_beams_list=[1]
        models=["meta_llama_Llama_3.1_8B_Instruct"]
        
    elif(test_name=="olmo_wmt"):
        mode = "wmt"
        suite_name="full_wmt_1_samples_1000_evals"
        num_beams_list=[2,4,8,16]
        models=["allenai_OLMo_2_1124_13B_Instruct"]
        
    
    elif(test_name=="olmo_gsm"):
        mode = "gsm"
        suite_name="full_wmt_1_samples_1000_evals"
        num_beams_list=[2,4,8]
        models=["allenai_OLMo_2_1124_13B_Instruct"]

    else:
        except_str=f"task name {test_name} not found"
        print(except_str)
        raise Exception(except_str)
    
    task_names, custom_metrics, instance_metrics= get_metrics(mode)
    return root_folder, num_beams_list, models, custom_metrics, task_names, suite_name, instance_metrics
    

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





def calculate_dict(init_dict, root_folder, num_beams_list:List[int], models:List[float], task_names:List[str], suite_name:str, dict_function)->Dict[int, GenerationSummary]:
    per_model=init_dict
    for model_idx, model in enumerate(models):        
        per_task={}
        for task_idx, task_name in enumerate(task_names):
            per_beam={}
            for num_beams in num_beams_list:
                run_folder=get_run_folder(root_folder=root_folder, num_beams=num_beams, model=model, task_name=task_name, suite_name=suite_name)
                obj=dict_function(run_folder)
                per_beam[num_beams]=obj
            per_task[task_idx]=per_beam
        per_model[model_idx] = per_task
    return per_model


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

def calculate_instances_dict(init_dict, root_folder, num_beams_list:List[int], models:List[float], task_names:List[str], suite_name:str):
    def get_instance_dict_from_run_folder(run_folder):
        gen_summary= get_gen_summary_from_run_folder(run_folder)
        instance_dict={}
        for instance_generation in gen_summary.instance_generations:
            instance_dict[instance_generation.instance_id] = instance_generation
        return instance_dict
    return calculate_dict(init_dict,root_folder, num_beams_list, models, task_names, suite_name,get_instance_dict_from_run_folder )


def get_metrics_dict(instances_dict:Dict[int, GenerationSummary], custom_metrics:List[PostMetric.PostMetric], instance_stats_dict):

    base_metrics=[PostMetric.TextMetric,PostMetric.SentenceLenMetric(),PostMetric.OutputProbMetric(),
                   PostMetric.InstanceIdMetric(), PostMetric.IsCompletionMetric()]
    metrics=base_metrics+custom_metrics
    metrics_dicts=[]   

    for model in instances_dict.keys():        
        for task_name in instances_dict[model].keys():
            for beam_num in instances_dict[model][task_name].keys():

                instance_stats_per_run = instance_stats_dict[model][task_name][beam_num]

                for instance_id, instance_generation in instances_dict[model][task_name][beam_num].items():
                    for example_idx,generated_output in enumerate(instance_generation.examples):
                        pd_metrics_dict=generated_output.stats_dict if generated_output.stats_dict  is not None else {} 
                        
                        pd_metrics_dict["beam_num"]=beam_num
                        pd_metrics_dict["task_name"]=task_name
                        pd_metrics_dict["model"]=model
                        pd_metrics_dict["example_idx"]=example_idx
                        
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
    metrics_dict:List[Dict[str,any]]=[]
    first_run_instances={}


    def __init__(self):
        self.instances_dict={}
        self.instance_stats_dict={}
        pass

    def init_with_mode(self, process_gens_modes:List[str]):
        print(f"Init: process_gens_mode {process_gens_modes}")
        if isinstance(process_gens_modes, str):
            process_gens_modes = [process_gens_modes]
        for process_gens_mode in process_gens_modes:
            
            root_folder, num_beams_list, models, custom_metrics, task_names, suite_name, instance_metrics= get_process_gen_params(process_gens_mode)
            self.init(root_folder=root_folder,num_beams_list=num_beams_list,models=models,custom_metrics=custom_metrics,task_names=task_names,  suite_name=suite_name,instance_metrics=instance_metrics)
            
    def get_params(self):
        root_folder     =self.process_gen_params["root_folder"]
        num_beams_list  =self.process_gen_params["num_beams_list"]
        models          =self.process_gen_params["models"]
        custom_metrics  =self.process_gen_params["custom_metrics"]
        task_names      =self.process_gen_params["task_names"]
        suite_name      =self.process_gen_params["suite_name"]
        instance_metrics=self.process_gen_params["instance_metrics"]
        return root_folder, num_beams_list, models, custom_metrics, task_names, suite_name, instance_metrics

    # def print_keys():
    #     firstkey=next(iter(processGens.instances_dict.keys()))
    #     print(firstkey)
    #     secondKey=next(iter(processGens.instances_dict[firstkey].keys()))
    #     print(secondKey)
    #     thirdKey=next(iter(processGens.instances_dict[firstkey][secondKey].keys()))
    #     print(thirdKey)
    #     fourthKey=next(iter(processGens.instances_dict[firstkey][secondKey][thirdKey].keys()))
    #     print(fourthKey)
    #     print("First prompt")
    #     print(processGens.instances_dict[firstkey][secondKey][thirdKey][fourthKey].prompt)

    #     print(processGens.instances_dict[0][0][2]["id10944"].prompt)


    def init(self,root_folder:str, num_beams_list:List[int], models:List[float], custom_metrics:List[PostMetric.PostMetric],task_names:List[str], instance_metrics:Dict[int, Dict[str, Dict[str, float]]]=None, suite_name:str=""):
        
        # #this is the pre-computed metrics
        # print("get_metrics_df")
        # self.metrics_df=self.get_metrics_df(root_folder)
        # print("get_instance_info")

        # #this is th
        # instance_info=self.get_instance_info(root_folder=root_folder, num_beams_list=num_beams_list, models=models,task_name= task_name,suite_name=suite_name)
        # self.instance_info=instance_info

        #get the generation summary for each task beam
        print("calculate_gen_summary_dict")
        
        
        self.instances_dict=calculate_instances_dict(init_dict=self.instances_dict, root_folder=root_folder, num_beams_list=num_beams_list, models=models,task_names=task_names, suite_name=suite_name)
        

        #get the run instance stats for each task and beam
        self.instance_stats_dict=calculate_instance_stats_dict(init_dict=self.instance_stats_dict, root_folder=root_folder, num_beams_list=num_beams_list, models=models, task_names=task_names, suite_name=suite_name, instance_metrics=instance_metrics)
        

        print("get_metrics_dict")
        self.metrics_dicts=get_metrics_dict(instances_dict=self.instances_dict, custom_metrics=custom_metrics, instance_stats_dict=self.instance_stats_dict)


        self.first_run_instances=get_first(get_first(get_first(self.instances_dict)))
        self.ids= list(self.first_run_instances.keys())


        self.process_gen_params = {"root_folder":root_folder ,
            "num_beams_list":num_beams_list,
            "models":models,
            "custom_metrics":custom_metrics,
            "task_names":task_names,
            "suite_name":suite_name,
            "instance_metrics":instance_metrics
        }
        


        
