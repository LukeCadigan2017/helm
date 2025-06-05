"""Run spec functions for HELM Instruct.

Website: https://crfm.stanford.edu/helm/instruct/"""

from typing import List

from helm.benchmark.adaptation.common_adapter_specs import get_instruct_adapter_spec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec

from helm.common.request import (BeamParams)


def get_instruction_following_critique_metric_specs(num_respondents: int) -> List[MetricSpec]:
    return []
        # MetricSpec(
        #     class_name="helm.benchmark.metrics.themis_instruction_following_critique_metrics.ThemisInstructionFollowingCritiqueMetric",
        #     # noqa E501
        #     args={"num_respondents": num_respondents},
        # )
        
    


@run_spec_function("self_instruct")
def get_self_instruct_spec(num_respondents: int, num_beams: int=1,num_return_sequences=1,top_p=1,top_k=0,temperature=1,batch_size=0, exact_mode_str="false") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.self_instruct_scenario.SelfInstructScenario",
        args={},
    )

    adapter_spec = get_instruct_adapter_spec(beam_params=BeamParams(num_beams=num_beams, num_return_sequences=num_return_sequences, top_p=top_p, top_k=top_k, temperature=temperature,batch_size=batch_size,exact_mode=(exact_mode_str=="true")), num_outputs=num_return_sequences)

    return RunSpec(
        name="self_instruct",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_instruction_following_critique_metric_specs(num_respondents),
        groups=["self_instruct"],
    )


@run_spec_function("vicuna")
def get_vicuna_spec(num_respondents: int, category: str = "all", num_beams: int=1,num_return_sequences=1,top_p=1,top_k=0,temperature=1,batch_size=0, exact_mode_str="false") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vicuna_scenario.VicunaScenario",
        args={"category": category},
    )

    adapter_spec = get_instruct_adapter_spec(beam_params=BeamParams(num_beams=num_beams, num_return_sequences=num_return_sequences, top_p=top_p, top_k=top_k, temperature=temperature,batch_size=batch_size,exact_mode=(exact_mode_str=="true")), num_outputs=num_return_sequences)

    return RunSpec(
        name=f"vicuna:category={category}",  # TODO: add args
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_instruction_following_critique_metric_specs(num_respondents),
        groups=["vicuna"],
    )


@run_spec_function("grammar")
def get_grammar_spec(num_respondents: int, path: str, tags: str, num_beams: int=1,num_return_sequences=1,top_p=1,top_k=0,temperature=1,batch_size=0, exact_mode_str="false") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.grammar_scenario.GrammarScenario",
        args={"path": path, "tags": tags},
    )

    adapter_spec = get_instruct_adapter_spec(beam_params=BeamParams(num_beams=num_beams, num_return_sequences=num_return_sequences, top_p=top_p, top_k=top_k, temperature=temperature,batch_size=batch_size,exact_mode=(exact_mode_str=="true")), num_outputs=num_return_sequences)

    return RunSpec(
        name=f"grammar:path={path},tags={tags}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_instruction_following_critique_metric_specs(num_respondents),
        groups=["grammar"],
    )


@run_spec_function("open_assistant")
def get_open_assistant_spec(num_respondents: int, language: str, num_beams: int=1,num_return_sequences=1,top_p=1,top_k=0,temperature=1,batch_size=0, exact_mode_str="false") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.open_assistant_scenario.OpenAssistantScenario",
        args={"language": language},
    )

    adapter_spec = get_instruct_adapter_spec(beam_params=BeamParams(num_beams=num_beams, num_return_sequences=num_return_sequences, top_p=top_p, top_k=top_k, temperature=temperature,batch_size=batch_size,exact_mode=(exact_mode_str=="true")), num_outputs=num_return_sequences)

    return RunSpec(
        name=f"open_assistant:language={language}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_instruction_following_critique_metric_specs(num_respondents),
        groups=["open_assistant"],
    )


@run_spec_function("koala")
def get_koala_spec(num_respondents: int, num_beams: int=1,num_return_sequences=1,top_p=1,top_k=0,temperature=1,batch_size=0, exact_mode_str="false") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.koala_scenario.KoalaScenario",
        args={},
    )

    adapter_spec = get_instruct_adapter_spec(beam_params=BeamParams(num_beams=num_beams, num_return_sequences=num_return_sequences, top_p=top_p, top_k=top_k, temperature=temperature,batch_size=batch_size,exact_mode=(exact_mode_str=="true")), num_outputs=num_return_sequences)

    return RunSpec(
        name="koala",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_instruction_following_critique_metric_specs(num_respondents),
        groups=["koala"],
    )


@run_spec_function("anthropic_hh_rlhf")
def get_anthropic_hh_rlhf_spec(num_respondents: int, subset: str, num_beams: int=1,num_return_sequences=1,top_p=1,top_k=0,temperature=1,batch_size=0, exact_mode_str="false") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.anthropic_hh_rlhf_scenario.AnthropicHHRLHFScenario",
        args={"subset": subset},
    )

    adapter_spec = get_instruct_adapter_spec(beam_params=BeamParams(num_beams=num_beams, num_return_sequences=num_return_sequences, top_p=top_p, top_k=top_k, temperature=temperature,batch_size=batch_size,exact_mode=(exact_mode_str=="true")), num_outputs=num_return_sequences)

    return RunSpec(
        name=f"anthropic_hh_rlhf:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_instruction_following_critique_metric_specs(num_respondents),
        groups=["anthropic_hh_rlhf"],
    )
