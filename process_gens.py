# import pandas as pd
# from helm.benchmark.runner import InstanceGenerations,GenerationSummary
# from typing import Any, List
# import json
# from helm.common.request import (GeneratedOutput)

# import ProcessGenMetrics
# import pandas as pd

# from typing import Dict


# def get_gen_summary(path) -> GenerationSummary:
#     def json_to_instance_generation(instance_dict:dict) -> InstanceGenerations:
#         instance_generation = InstanceGenerations(**instance_dict)
#         examples = [ GeneratedOutput(**generated_output_dict) for generated_output_dict in instance_generation.examples]
#         instance_generation.examples=examples
#         return instance_generation
#     with open(path,'r') as json_file:
#         generation_summary_dict=json.load(json_file)
#     generation_summary=GenerationSummary(**generation_summary_dict)
#     instance_generations = [json_to_instance_generation(instance_dict)  for instance_dict in generation_summary.instance_generations ]
#     generation_summary.instance_generations=instance_generations
#     return generation_summary

# def ProcessGens():
#     base_folder:str
#     beam_num_to_summary:Dict[int, GenerationSummary]
#     metrics_dict:List[Dict[str,any]]

#     def __init__(self,base_folder:str, num_beams_list:List[int], models:List[float], custom_metrics:List[ProcessGenMetrics.PostMetric]):
#         metrics_df=self.get_metrics_df(base_folder)
       
#         beam_num_to_summary=self.calculate_beam_num_to_summary(base_folder, num_beams_list, models)
#         metrics_dicts=self.get_metrics_dict(beam_num_to_summary, custom_metrics)
#         id_to_metrics_dicts=self.get_id_to_metrics(metrics_dicts)
        
#         self.base_folder=base_folder
#         self.metrics_df=metrics_df

#         self.beam_num_to_summary=beam_num_to_summary
#         self.metrics_dicts=metrics_dicts
#         self.id_to_metrics_dicts=id_to_metrics_dicts


#     @classmethod  
#     def calculate_beam_num_to_summary(self, base_folder, num_beams_list:List[int], models:List[float])->Dict[int, GenerationSummary]:
#         beam_num_to_summary= {}
#         for num_beams in num_beams_list:
#             for model in models:
#                 path=f"{base_folder}/wmt/{model}/{num_beams}_beams/runs/eval_600/generation_summary.json"
#                 generation_summary=get_gen_summary(path)
#                 beam_num_to_summary[num_beams]=generation_summary
#         return beam_num_to_summary[num_beams]

#     @classmethod  
#     def get_metrics_dict(self, beam_num_to_summary:Dict[int, GenerationSummary], custom_metrics:List[ProcessGenMetrics.PostMetric]):
#         base_metrics=[ProcessGenMetrics.SentenceLenMetric(),ProcessGenMetrics.CompletionProbMetric(), ProcessGenMetrics.InstanceIdMetric(), ProcessGenMetrics.IsCompletionMetric()]
#         metrics=base_metrics+custom_metrics
#         metrics_dicts=[]
#         for beam_num, generation_summary in self.beam_num_to_summary.items():
#             print(beam_num)
#             for instance_generation in generation_summary.instance_generations:
#                 for generated_output in instance_generation.examples:
#                     pd_metrics_dict={}
#                     for metric in metrics:
#                         pd_metrics_dict[metric.name()] = metric.calculate_metric(instance_generation, generated_output)
#                     pd_metrics_dict["beam_num"]=beam_num
#                     metrics_dicts.append(pd_metrics_dict)
#         return metrics_dicts

    

#     @classmethod  
#     def get_id_to_metrics(self, metrics_dicts):
#         id_to_metrics_dicts={}
#         for metrics_dict in metrics_dicts:
#             instanceID=ProcessGenMetrics.get_post_metric(metrics_dict=metrics_dict,metric_type=ProcessGenMetrics.InstanceIdMetric)
#             beam_num=ProcessGenMetrics.get_post_metric(metrics_dict=metrics_dict,metric_type=ProcessGenMetrics.BeamNumMetric)
#             if instanceID not in id_to_metrics_dicts.keys():
#                 id_to_metrics_dicts[instanceID]={}
#             id_to_metrics_dicts[instanceID][beam_num]=metrics_dict
#         return id_to_metrics_dicts

#     @classmethod  
#     def get_metrics_df(self, base_folder):
#         metrics_file=f"{base_folder}/metrics_csv.txt"
#         raw_metric_df = pd.read_csv(metrics_file, header=None)
#         raw_metric_df.columns=[ "model", "task", "beam_num", "metric", "value"]
#         raw_metric_df.drop(["task"],axis=1)
#         metric_df = raw_metric_df.pivot(
#             index=["model", "beam_num"],
#             columns="metric",
#             values="value"
#         ).reset_index()
#         metric_df.sort_values("beam_num")
#         return metric_df
    


