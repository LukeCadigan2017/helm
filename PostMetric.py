import pandas as pd
from helm.benchmark.runner import InstanceGenerations,GenerationSummary
from typing import Any, List
import json
from helm.common.request import (GeneratedOutput)

import torch
from typing import Dict




from dataclasses import dataclass

from abc import abstractmethod, ABC

from helm.benchmark.metrics.comet_metric import CometMetric


from helm.common.gpu_utils import get_torch_device_name

############################ General ############################

def get_post_metric(metrics_dict, metric_type):
    return metrics_dict[metric_type.name()]

    # calculate_post_metric(metrics_dict=pd_metrics_dict,metric=metric, instance_generation=instance_generation,generated_output=generated_output)
def calculate_post_metric(metrics_dict,metric,  instance_generation,generated_output):
    metrics_dict[metric.name()] = metric.calculate_metric(instance_generation, generated_output)
    return metrics_dict



############################ Abstract Class ############################

class PostMetric(ABC):
    @property
    @abstractmethod
    def name(self)->str:
        pass
    @abstractmethod
    def calculate_metric(self, instance_generation:InstanceGenerations,generated_output:GeneratedOutput) -> Any:
        pass

############################ Test Metrics ############################

class TestMetric(PostMetric):
    @classmethod
    def name(cls)->str:
        return "test"
    @classmethod    
    def calculate_metric(self,instance_generation:InstanceGenerations,generated_output:GeneratedOutput) -> float:
        return 0

class Test2Metric(PostMetric):
    @classmethod
    def name(cls)->str:
        return "test2"
    @classmethod    
    def calculate_metric(self,instance_generation:InstanceGenerations,generated_output:GeneratedOutput) -> float:
        return 1


############################ Snellius Metrics ############################

# class CometPostMetric(PostMetric):
#     @classmethod
#     def name(cls)->str:
#         return "comet"
#     @classmethod    
#     def calculate_metric(self,instance_generation:InstanceGenerations,generated_output:GeneratedOutput) -> float:


#         comet_metric = CometMetric(task="task", device=get_torch_device_name())
#         ref = instance_generation.reference.strip()
#         src = instance_generation.prompt.strip()
#         mt = generated_output.text.strip()
#         comet_metric.evaluate_generation(ref=ref,src=src,mt=mt)


        # """Compute the COMET score for this instance"""
        # ref = instance_generation.reference.strip()
        # src = instance_generation.prompt.strip()
        # mt = generated_output.text.strip()

        # # comet requires this exact format
        # data = [dict(ref=ref, src=src, mt=mt)]
        # output = self.comet_scorer.predict(data, gpus=self.num_gpus, progress_bar=False)  # type: ignore
        # comet_score = output[0][0]  # extract the actual score
        # return comet_score


############################ Base Example Metrics ############################

class TextMetric(PostMetric):
    @classmethod
    def name(cls)->str:
        return "text"
    @classmethod    
    def calculate_metric(self,instance_generation:InstanceGenerations,generated_output:GeneratedOutput) -> str:
        return generated_output.text

class SentenceLenMetric(PostMetric):
    @classmethod
    def name(cls)->str:
        return "completion_length"
    @classmethod    
    def calculate_metric(self,instance_generation:InstanceGenerations,generated_output:GeneratedOutput) -> float:
        return len(generated_output.text)
    
class OutputProbMetric(PostMetric):
    @classmethod
    def name(cls)->str:
        return "output_logprob"
    @classmethod    
    def calculate_metric(self,instance_generation:InstanceGenerations,generated_output:GeneratedOutput) -> float:
        return generated_output.logprob

class BeamNumMetric(PostMetric):
    @classmethod
    def name(cls)->str:
        return "beam_num"
    @classmethod    
    def calculate_metric(self,instance_generation:InstanceGenerations,generated_output:GeneratedOutput) -> float:
        return instance_generation.beam_num

class ModelMetric(PostMetric):
    @classmethod
    def name(cls)->str:
        return "model"
    @classmethod    
    def calculate_metric(self,instance_generation:InstanceGenerations,generated_output:GeneratedOutput) -> float:
        return instance_generation.model
    
class InstanceIdMetric(PostMetric):
    @classmethod
    def name(cls)->str:
        return "instanceID"
    @classmethod
    def calculate_metric(self, instance_generation:InstanceGenerations,generated_output:GeneratedOutput) -> str:
        return instance_generation.instance_id






class IsCompletionMetric(PostMetric):
    @classmethod
    def name(cls)->str:
        return "isCompletion"
    @classmethod
    def calculate_metric(self,instance_generation:InstanceGenerations,generated_output:GeneratedOutput) -> float:
        pass
        # return generated_output.text == instance_generation.completion 
    
############################ Special Example Metrics ############################
from helm.benchmark.metrics.evaluate_reference_metrics import bleu_4, bleu_1

class BLEU4_METRIC(PostMetric):
    @classmethod
    def name(cls)->str:
        return "BLEU_4"
    @classmethod
    def calculate_metric(self,instance_generation:InstanceGenerations,generated_output:GeneratedOutput) -> float:
        return bleu_4(generated_output.text, instance_generation.reference)     


class BLEU1_METRIC(PostMetric):
    @classmethod
    def name(cls)->str:
        return "BLEU_1"
    @classmethod
    def calculate_metric(self,instance_generation:InstanceGenerations,generated_output:GeneratedOutput) -> float:
        return bleu_1(generated_output.text, instance_generation.reference)     


############################ INSTANCE METRICS ############################

class ReferenceMetric(PostMetric):
    @classmethod
    def name(cls)->str:
        return "reference"
    @classmethod    
    def calculate_metric(self,instance_generation:InstanceGenerations,generated_output:GeneratedOutput) -> str:
        return instance_generation.reference
    
class InstanceCompletionMetric(PostMetric):
    @classmethod
    def name(cls)->str:
        return "instance_completion"
    
    @classmethod    
    def calculate_metric(self,instance_generation:InstanceGenerations,generated_output:GeneratedOutput) -> str:
        return instance_generation.completion
    
class InstanceCompletionMetric(PostMetric):
    @classmethod
    def name(cls)->str:
        return "instance_completion"
    
    @classmethod    
    def calculate_metric(self,instance_generation:InstanceGenerations,generated_output:GeneratedOutput) -> str:
        return instance_generation.completion


#note: metric must be in all metrics for this to work
all_metrics=[InstanceCompletionMetric, ReferenceMetric, BLEU1_METRIC, BLEU4_METRIC, IsCompletionMetric, InstanceIdMetric, ModelMetric, BeamNumMetric, OutputProbMetric, SentenceLenMetric, TextMetric, CometPostMetric, TestMetric, Test2Metric]
def get_post_metrics(special_metric_names):
    special_metrics=[]
    for metric in all_metrics:
        metric_name=(metric()).name() 
        if( metric_name in special_metric_names):
            special_metrics.append(metric())
    assert(len(special_metric_names)== len(special_metrics))
    return special_metrics