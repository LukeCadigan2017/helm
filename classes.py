import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from helm.common.media_object import MultimediaObject
from helm.common.image_generation_parameters import ImageGenerationParameters
from helm.common.general import indent_lines, format_text
import dacite
import json
import math
import os
import traceback
import typing
from collections import Counter
import dataclasses
from typing import Any, Dict, List
import numpy as np
import datetime
import pprint

from tqdm import tqdm
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.common.request import (GeneratedOutput)
from helm.common.general import ensure_directory_exists, write, asdict_without_nones
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.cache import cache_stats
from helm.benchmark.scenarios.scenario import (
    EVAL_SPLITS,
    TRAIN_SPLIT,
    Scenario,
    create_scenario,
    Instance,
    get_scenario_cache_path,
    with_instance_ids,
)
from helm.benchmark.adaptation.adapters.adapter import Adapter
from helm.benchmark.adaptation.adapters.adapter_factory import AdapterFactory
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.run_spec import RunSpec
from helm.benchmark.data_preprocessor import DataPreprocessor
from helm.benchmark.executor import ExecutionSpec, Executor
from helm.benchmark.annotation_executor import AnnotationExecutionSpec, AnnotationExecutor
from helm.benchmark.metrics.dry_run_metrics import DryRunMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.metric import MetricInterface, MetricResult, PerInstanceStats, create_metric, Stat
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from dataclasses import dataclass




@dataclass(frozen=True)
class Token:
    """
    A `Token` represents one token position in a `Sequence`, which has the
    chosen `text` as well as the top probabilities under the model.
    """

    # Text that was chosen
    text: str

    # Log probability of generating that
    logprob: float

    token_id: int = None
    def render_lines(self) -> List[str]:
        return [
            f"{format_text(self.text)} logprob={self.logprob}",
        ]
    

class GeneratedOutput:
    """A `GeneratedOutput` is a single generated output that may contain text or multimodal content."""

    # The concatenation of all the tokens
    text: str

    # The sum of the log probabilities of all tokens
    logprob: float

    # The tokens
    tokens: List[Token]

    # Why did the sequence finish?
    finish_reason: Optional[Dict[str, Any]] = None

    # Could be a sequence made up of multimedia content
    multimodal_content: Optional[MultimediaObject] = None

    #before concatenation
    full_text: str=None


@dataclass(frozen=False)
class InstanceGenerations:
    """Split (e.g., train, valid, test)"""
    instance_id: str
    """id of instance"""

    prompt: str
    """Prompt used"""
    
    completion: str
    """Selection completion for metrics"""

    # The sum of the log probabilities of all tokens
    completion_logprob: float
    """Completion probability"""

    full_prompt: str

    examples: List[GeneratedOutput]
    """List of unscored examples"""

    reference: str=None
    """Reference used"""
    beam_num: int=None
    model: str=None


@dataclass(frozen=False)
class GenerationSummary:
    task_name:str
    instance_generations :List[InstanceGenerations]
    adapter_spec: AdapterSpec
