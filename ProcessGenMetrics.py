import pandas as pd
from helm.benchmark.runner import InstanceGenerations,GenerationSummary
from typing import Any, List
import json
from helm.common.request import (GeneratedOutput)

from typing import Dict




from dataclasses import dataclass

from abc import abstractmethod, ABC


############################ General ############################

def get_post_metric(metrics_dict, metric_type):
    return metrics_dict[metric_type.name()]

    # calculate_post_metric(metrics_dict=pd_metrics_dict,metric=metric, instance_generation=instance_generation,generated_output=generated_output)
def calculate_post_metric(metrics_dict,metric,  instance_generation,generated_output):
    metrics_dict[metric.name()] = metric.calculate_metric(instance_generation, generated_output)
    return metrics_dict



class PostMetric(ABC):
    @property
    @abstractmethod
    def name(self)->str:
        pass
    @abstractmethod
    def calculate_metric(self, instance_generation:InstanceGenerations,generated_output:GeneratedOutput) -> Any:
        pass

class TestMetric(PostMetric):
    @classmethod
    def name(cls)->str:
        return "test"
    @classmethod    
    def calculate_metric(self,instance_generation:InstanceGenerations,generated_output:GeneratedOutput) -> float:
        return 0


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

