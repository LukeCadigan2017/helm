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



def get_gen_summary_from_path(path) -> GenerationSummary:
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
    with open(path,'r') as json_file:
        generation_summary_dict=json.load(json_file)
    generation_summary=GenerationSummary(**generation_summary_dict)
    instance_generations = [json_to_instance_generation(instance_dict)  for instance_dict in generation_summary.instance_generations ]
    generation_summary.instance_generations=instance_generations
    return generation_summary

def get_run_folder(base_folder, num_beams:int, model:str, task_name: str, eval_instances:int):
    return f"{base_folder}/{task_name}/{model}/{num_beams}_beams/runs/eval_{eval_instances}"

def get_gen_summary(base_folder, num_beams:int, model:str, task_name: str, eval_instances:int):
    path=get_run_folder(base_folder, num_beams, model, task_name, eval_instances)+"/generation_summary.json"
    return get_gen_summary_from_path(path)

def truncate_sequence(text:str, all_stops=["<|end_of_text|>"]) -> str:
    for stop in all_stops:
        try:
            text = text[: text.index(stop)]
        except ValueError:
            pass
    return text.strip()

class ProcessGens:
    base_folder:str
    beam_num_to_summary:Dict[int, GenerationSummary]
    metrics_dict:List[Dict[str,any]]

    def __init__(self,base_folder:str, num_beams_list:List[int], models:List[float], custom_metrics:List[PostMetric.PostMetric],task_name:str, eval_instances:str=None, instance_metrics:Dict[int, Dict[str, Dict[str, float]]]=None):
        

        self.base_folder=base_folder

        # #these are by themselves
        print("get_metrics_df")
        metrics_df=self.get_metrics_df(base_folder)
        print("get_instance_info")
        instance_info=self.get_instance_info(base_folder=base_folder, num_beams_list=num_beams_list, models=models,task_name= task_name, eval_instances=eval_instances)


        beam_num_to_instance_stats=self.calculate_beam_num_to_run_instance_stats(base_folder=base_folder, num_beams_list=num_beams_list, models=models, task_name=task_name, eval_instances=eval_instances, instance_metrics=instance_metrics)
        # self.metrics_df=metrics_df
        self.instance_info=instance_info

        #these go together
        print("calculate_beam_num_to_summary")
        beam_num_to_summary=self.calculate_beam_num_to_summary(base_folder=base_folder, num_beams_list=num_beams_list, models=models,eval_instances=eval_instances,task_name=task_name)
        print("get_metrics_dict")
        metrics_dicts=self.get_metrics_dict(beam_num_to_summary=beam_num_to_summary, custom_metrics=custom_metrics, beam_num_to_instance_stats=beam_num_to_instance_stats)

        self.beam_num_to_instance_stats=beam_num_to_instance_stats
        self.beam_num_to_summary=beam_num_to_summary
        self.metrics_dicts=metrics_dicts
        self.metrics_df=metrics_df

    
    def calculate_beam_num_to_run_instance_stats(self, base_folder, num_beams_list:List[int], models:List[float], task_name:str, eval_instances:int, instance_metrics:Dict[int, Dict[str, Dict[str, float]]])->Dict[int, List[PerInstanceStats]]:
        def json_to_run_instance_stats(path, instance_metrics) -> List[PerInstanceStats]:
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
        
        beam_num_to_run_instance_stats= {}
        for num_beams in num_beams_list:
            for model in models:
                path=f"{base_folder}/{task_name}/{model}/{num_beams}_beams/runs/eval_{eval_instances}/per_instance_stats.json"
                print(f"Analyzing path: {path}")
                run_instance_stats = json_to_run_instance_stats(path, instance_metrics)
                beam_num_to_run_instance_stats[num_beams]=run_instance_stats
        return beam_num_to_run_instance_stats

    def clean_generation_summary(generationSummary:GenerationSummary, eval_instances)->GenerationSummary:
        def clean_instance_generation(instanceGenerations:InstanceGenerations)->InstanceGenerations:
            def clean_generated_output(generatedOutput:GeneratedOutput)-> GeneratedOutput:
                generatedOutput.text=truncate_sequence(generatedOutput.text)
                return generatedOutput
            instanceGenerations.examples=[clean_generated_output(example) for example in instanceGenerations.examples]
            instanceGenerations.examples.sort(key=lambda x:float(x.logprob),reverse=True)
            completion=instanceGenerations.examples[0]
            instanceGenerations.completion=completion.text
            instanceGenerations.completion_logprob=completion.logprob
            return instanceGenerations
        generationSummary.instance_generations=[clean_instance_generation(instance_generation) for instance_generation in generationSummary.instance_generations]
        assert len(generationSummary.instance_generations)==eval_instances
        print(f"number of instances: {len(generationSummary.instance_generations)}")
        generationSummary.instance_generations=generationSummary.instance_generations[:eval_instances]
        return generationSummary

        


    @classmethod  
    def get_instance_info(self, base_folder, num_beams_list:List[int], models:List[str], task_name: str, eval_instances:int)->Dict[int, GenerationSummary]:
        num_beams=num_beams_list[0]
        model=models[0]
        instance_infos= {}
        instance_metrics=[PostMetric.ReferenceMetric()]

        generation_summary=get_gen_summary(base_folder, num_beams, model, task_name, eval_instances)
        for instance_generation in generation_summary.instance_generations:
            instance_id=instance_generation.instance_id
            if instance_id not in instance_infos.keys():
                instance_dict={}
                for metric in instance_metrics:
                    instance_dict=PostMetric.calculate_post_metric(instance_dict,metric,instance_generation,None)
                instance_infos[instance_id]=instance_dict
        return instance_infos

    @classmethod  
    def calculate_beam_num_to_summary(self, base_folder, num_beams_list:List[int], models:List[float], task_name: str,eval_instances:int)->Dict[int, GenerationSummary]:
        beam_num_to_summary= {}
        for num_beams in num_beams_list:
            for model in models:
                # path=f"{base_folder}/{task_name}/{model}/{num_beams}_beams/runs/eval_{eval_instances}/generation_summary.json"
                # print(f"Analyzing path: {path}")
                # raw_generation_summary=get_gen_summary(path)
                raw_generation_summary=get_gen_summary(base_folder, num_beams, model, task_name, eval_instances)
                generation_summary:GenerationSummary=self.clean_generation_summary(raw_generation_summary, eval_instances)
                # if(eval_instances):
                    # assert len(generation_summary.instance_generations)==eval_instances
                beam_num_to_summary[num_beams]=generation_summary
        return beam_num_to_summary

    @classmethod  
    def get_metrics_dict(self, beam_num_to_summary:Dict[int, GenerationSummary], custom_metrics:List[PostMetric.PostMetric], beam_num_to_instance_stats: Dict[int, Dict[str, Dict[str, float]]]):
        base_metrics=[PostMetric.TextMetric,PostMetric.SentenceLenMetric(),PostMetric.OutputProbMetric(),
                       PostMetric.InstanceIdMetric(), PostMetric.IsCompletionMetric()]
        metrics=base_metrics+custom_metrics
        metrics_dicts=[]
        for beam_num, generation_summary in beam_num_to_summary.items():
            print(beam_num)
            for instance_generation in generation_summary.instance_generations:
                for idx,generated_output in enumerate(instance_generation.examples):
                    pd_metrics_dict={}
                    for metric in metrics:
                        pd_metrics_dict=PostMetric.calculate_post_metric(pd_metrics_dict,metric,instance_generation,generated_output)
                    pd_metrics_dict["beam_num"]=beam_num
                    if(idx==0):
                        pd_metrics_dict["isCompletion"]=(idx==0)
                        completion_metrics_dict = beam_num_to_instance_stats[beam_num][instance_generation.instance_id]
                        for stat_name, value in completion_metrics_dict.items():
                            pd_metrics_dict[stat_name]= value
                    metrics_dicts.append(pd_metrics_dict)
        return metrics_dicts

    @classmethod  
    def get_metrics_df(self, base_folder):

        try:
            metrics_file=f"{base_folder}/metrics_csv.txt"
            raw_metric_df = pd.read_csv(metrics_file, header=None)
            raw_metric_df.columns=[ "model", "task", "beam_num", "metric", "value"]
            raw_metric_df.drop(["task"],axis=1)
            metric_df = raw_metric_df.pivot(
                index=["model", "beam_num"],
                columns="metric",
                values="value"
            ).reset_index()
            metric_df.sort_values("beam_num")
            self.metric_df=metric_df
            return metric_df
        except:
            return None
