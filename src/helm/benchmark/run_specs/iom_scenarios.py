# from helm.benchmark.adaptation.adapter_spec import (
#     ADAPT_GENERATION,
#     ADAPT_MULTIPLE_CHOICE_JOINT,
#     AdapterSpec,
# )
# from helm.benchmark.adaptation.common_adapter_specs import (
#     get_generation_adapter_spec,
# )
# from helm.benchmark.metrics.common_metric_specs import (
#     get_basic_generation_metric_specs,
#     get_generative_harms_metric_specs,
#     get_generic_metric_specs
# )
# from helm.benchmark.run_spec import RunSpec, run_spec_function
# from helm.benchmark.runner import get_benchmark_output_path
# from helm.benchmark.scenarios.scenario import ScenarioSpec, get_scenario_cache_path


# @run_spec_function("gsm_iom")
# def get_gsm_iom_spec(num_beams: int=1) -> RunSpec:

#     scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.gsm_scenario.GSM8KScenario", args={})

#     # Create AdapterSpec based on the GSM8K paper: https://arxiv.org/pdf/2110.14168.pdf
#     adapter_spec = get_generation_adapter_spec(
#         input_noun="Q",
#         output_noun="A",
#         max_train_instances=5,  # Due to limited context and long example length
#         max_tokens=400,  # The paper uses 400 tokens as the max sample length
#         stop_sequences=["\n\n"],  # Since answer may contain newlines, we use two as SEP
#         num_beams=num_beams
#     )

#     return RunSpec(
#         name="gsm_iom",
#         scenario_spec=scenario_spec,
#         adapter_spec=adapter_spec,
#         metric_specs=get_basic_generation_metric_specs(["exact_match_indicator", "final_number_exact_match"])
#         + get_generic_metric_specs()
#         + get_generative_harms_metric_specs(),
#         groups=["gsm_iom"],
#     )


# @run_spec_function("gsm_iom2")
# def get_gsm_iom_spec2(num_beams: int=1) -> RunSpec:

#     scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.gsm_scenario.GSM8KScenario", args={})

#     # Create AdapterSpec based on the GSM8K paper: https://arxiv.org/pdf/2110.14168.pdf
#     adapter_spec = get_generation_adapter_spec(
#         input_noun="Q",
#         output_noun="A",
#         max_train_instances=5,  # Due to limited context and long example length
#         max_tokens=400,  # The paper uses 400 tokens as the max sample length
#         stop_sequences=["\n\n"],  # Since answer may contain newlines, we use two as SEP
#         num_beams=num_beams
#     )

#     return RunSpec(
#         name="gsm_iom",
#         scenario_spec=scenario_spec,
#         adapter_spec=adapter_spec,
#         metric_specs=get_basic_generation_metric_specs(["exact_match_indicator", "final_number_exact_match"])
#         + get_generic_metric_specs()
#         + get_generative_harms_metric_specs(),
#         groups=["gsm_iom"],
#     )
