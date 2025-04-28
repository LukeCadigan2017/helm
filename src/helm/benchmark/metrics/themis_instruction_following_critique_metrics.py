# noqa: E501
from typing import Dict, List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.critique_request import CritiqueTaskTemplate, CritiqueRequest, QuestionType,CritiqueQuestionTemplate


class ThemisInstructionFollowingCritiqueMetric(Metric):
    """
    Critique evaluation for instruction following. Possesses the ability to ask human
    annotators the following questions about the model responses:

    1. Response relevance/helpfulness
    2. How easy it is to understand the response
    3. How complete the response is
    4. How concise the response is
    5. Whether the response uses toxic language or helps the user with harmful goals
    6. Whether all facts cited in the response are true
    """

    # "task": "Instruction Following",  # Which NLG task does the sample belongs to, e.g. Summarization,
    #   "source_des": "Instruction",  # The description of the source, e.g. Article
    #   "target_des": "Response",  # The description of the target, e.g. Summary
    #   "aspects": []

    def __init__(self, num_respondents: int) -> None:
        self._template = CritiqueTaskTemplate(
            name="themis_instruction_following_critique",
            instructions="""###Instruction###
Please act as an impartial and helpful evaluator for natural language generation (NLG), and the audience is an
expert in the field.
Your task is to evaluate the quality of Instruction Following strictly based on the given evaluation criterion.
Begin the evaluation by providing your analysis concisely and accurately, and then on the next line, start with
"Rating:" followed by your rating on a Likert scale from 1 to 5 (higher means better).
You MUST keep to the strict boundaries of the evaluation criterion and focus solely on the issues and errors
involved; otherwise, you will be penalized.
Make sure you read and understand these instructions, as well as the following evaluation criterion and example
content, carefully.
###Evaluation Criterion###
Overall Quality
###Example###
Instruction:
{{instruction}}
Response:
{{response}}
###Your Evaluation###
""",
            num_respondents=num_respondents,
            questions=[
            CritiqueQuestionTemplate(
                    name="Overall Quality",
                    question_type=QuestionType.FREE_RESPONSE,
                    # Note: Text can contain HTML.
                    text="",
                    options=[],
                )]
    )

    def __repr__(self) -> str:
        return "InstructionFollowingCritiqueMetric()"

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """Get critiques of a summary and compute metrics based on the critiques."""
        assert request_state.result is not None
        if len(request_state.result.completions) != 1:
            raise ValueError("InstructionFollowingCritiqueMetric only supports a single generation per instance")
        model_response: str = request_state.result.completions[0].text
        request = CritiqueRequest(
            self._template, fields={"instruction": request_state.instance.input.text, "response": model_response}
        )
        result = metric_service.make_critique_request(request)
        if not result or not result.responses:
            return []
        stats: Dict[str, Stat] = {}
        for question in self._template.questions:
            stats[question.name] = Stat(MetricName(question.name))
        # Skip computing metrics if there are not enough responses.
        if len(result.responses) < request.template.num_respondents:
            return []
        for response in result.responses:
            for answer_name, answer in response.answers.items():
                if not isinstance(answer, str):
                    raise ValueError(f"Expected answer to {answer_name} be a string")
                answer_value: float = 0
                stats[answer_name].add(answer_value)
        return list(stats.values())
