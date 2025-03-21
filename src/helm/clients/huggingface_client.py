from copy import deepcopy
import torch
from transformers import AutoModelForCausalLM
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing import Any, Dict, List, Optional, TypedDict

from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import htrack_block, hlog
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import (
    wrap_request_time,
    EMBEDDING_UNAVAILABLE_REQUEST_RESULT,
    Request,
    RequestResult,
    GeneratedOutput,
    Token,
)
from helm.tokenizers.tokenizer import Tokenizer
from helm.clients.client import CachingClient, truncate_sequence
from helm.tokenizers.huggingface_tokenizer import HuggingFaceTokenizer, WrappedPreTrainedTokenizer
from threading import Lock
import json
from pprint import pprint

class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, stop_sequence: List[int]):
        super().__init__()
        self.stop_sequence = stop_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Create a tensor from the stop_sequence
        stop_sequence_tensor = torch.tensor(self.stop_sequence, device=input_ids.device, dtype=input_ids.dtype)

        # Check if the current sequence ends with the stop_sequence
        current_sequence = input_ids[:, -len(self.stop_sequence) :]
        return bool(torch.all(current_sequence == stop_sequence_tensor).item())

# class StopOnStrings(StoppingCriteria):
#     def __init__(self, stop_strings, tokenizer):
#         self.stop_strings = stop_strings
#         self.tokenizer = tokenizer
#         self.stream = ""

#     def reset(self):
#         self.stream = ""

#     def __call__(self, input_ids, scores, **kwargs):
#         # print("input_ids shape is ",input_ids.size())
#         generated = self.tokenizer.decode(input_ids[0][-1], skip_special_tokens=True)
#         self.stream += generated
#         for stop_string in self.stop_strings:
#             # print("Hello! stop_string is ",stop_string)
#             # print("Hello! steam is is ",self.stream)
#             if self.stream[:-1].endswith(stop_string):
#                 # print("Hello! got here")
#                 return True
#         # self.stream = ""
#         # print(generated, end="", flush=True)
#         return False



class HuggingFaceRequest(TypedDict):
    """Data passed between make_request and serve_request. Used as the cache key."""

    engine: str
    prompt: str
    temperature: float
    num_beams:int
    generated_output_file: str
    num_return_sequences: int
    max_new_tokens: int
    top_p: float
    echo_prompt: bool
    top_k_per_token: int
    stop_sequences: List


class HuggingFaceServer:
    """A thin wrapper around a Hugging Face AutoModelForCausalLM for HuggingFaceClient to call."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        wrapped_tokenizer: WrappedPreTrainedTokenizer,
        **kwargs,
    ):
        self._lock= Lock()
        self.stop_sequence_dict={}
        self.device: Optional[str]
        if "device_map" in kwargs:
            if "device" in kwargs:
                raise ValueError("At most one of one of `device` and `device_map` may be specified.")
            try:
                import accelerate  # noqa: F401
            except ModuleNotFoundError as e:
                handle_module_not_found_error(e, ["accelerate"])
            hlog(f'Hugging Face device_map set to "{kwargs["device_map"]}" from kwargs.')
            self.device = None
        elif "device" in kwargs:
            if "device_map" in kwargs:
                raise ValueError("At most one of one of `device` and `device_map` may be specified.")
            hlog(f'Hugging Face device set to "{kwargs["device"]}" from kwargs.')
            self.device = kwargs.pop("device")
        elif torch.cuda.is_available():
            hlog('Hugging Face device set to "cuda:0" because CUDA is available.')
            self.device = "cuda:0"
        else:
            hlog('Hugging Face device set to "cpu" because CUDA is unavailable.')
            self.device = "cpu"

        # Security issue: currently we trust remote code by default.
        # We retain this temporarily to maintain reverse compatibility.
        # TODO: Delete if-else and don't set trust_remote_code=True
        if "trust_remote_code" not in kwargs:
            kwargs["trust_remote_code"] = True

        with htrack_block(f"Loading Hugging Face model {pretrained_model_name_or_path}"):
            # WARNING this may fail if your GPU does not have enough memory
            if self.device is None:
                # kwargs contains device_map=auto
                # Do not call to() because accelerate will take care of model device placement.
                self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs).to(
                    self.device
                )
        self.wrapped_tokenizer = wrapped_tokenizer

    def serve_request(self, raw_request: HuggingFaceRequest) -> Dict:
        with self.wrapped_tokenizer as tokenizer:
            encoded_input = tokenizer(raw_request["prompt"], return_tensors="pt", return_token_type_ids=False).to(
                0 if self.device is None else self.device
            )
        stopping_criteria: Optional[StoppingCriteriaList] = None
        optional_args = {}
        if len(raw_request["stop_sequences"]) > 0:
            stopping_criteria = StoppingCriteriaList()
            raise Exception("Not implemented!!")
            # stopping_criteria.append(StopOnStrings(raw_request["stop_sequences"], tokenizer))

        # Check if we need to compute the perplexity of the prompt (#1497)
        compute_logprobs_only = (
            raw_request["max_new_tokens"] == 0
            and raw_request["num_return_sequences"] == 1
            and raw_request["echo_prompt"]
        )

        num_generated=max(raw_request["num_return_sequences"], raw_request["num_beams"])
        assert(raw_request["top_p"]==1)


        # Use HuggingFace's `generate` method.
        if compute_logprobs_only:
            with torch.no_grad():
                output = self.model(encoded_input["input_ids"])
            sequences = encoded_input["input_ids"]
            scores = output.logits
        else:
            # print(f"stopping_criteria is {stopping_criteria}")
            # print(f"optional_args is {optional_args}")
            output = self.model.generate(
                **encoded_input,
                length_penalty=0,
                # temperature=raw_request["temperature"],
                num_beams = raw_request["num_beams"],
                num_return_sequences=num_generated,
                max_new_tokens=raw_request["max_new_tokens"],
                # top_p=raw_request["top_p"],
                #changed this
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                **optional_args,
                stopping_criteria=stopping_criteria, 
            )
            sequences = output.sequences
            scores = output.scores

        eos_token=sequences[0][-1]
        # print("\n\n\n\n eos token is ",eos_token)
        prompt_tokens_logprobs = []
        # if compute_logprobs_only:
        #     # Append the logprob of the first token of the prompt.
        #     prompt_tokens_logprobs.append(0.0)

        #     # Compute logprobs of prompt tokens.
        #     for completion_id in range(num_generated):
        #         for i in range(len(sequences[completion_id]) - 1):
        #             logprobs = torch.nn.functional.log_softmax(scores[completion_id][i], dim=0)
        #             prompt_tokens_logprobs.append(logprobs[sequences[completion_id][i + 1]].item())


        # for completion_id in range(num_generated):
        #     print("len[sequences] is ",len(sequences[completion_id]))

        # input_end=len(encoded_input.input_ids[0])

        # score_len=len(scores)
        # for idx, sequence in enumerate(sequences):
        #     assert (len(encoded_input.input_ids[0])+score_len)==len(sequence), f"Input len: {len(encoded_input.input_ids[0])}, score len {score_len}, sequence len {len(sequence)}, idx {idx}"
        
        # sequences = [sequence[-score_len:] for sequence in sequences]
        


        ########## THEIR ATTEMPT ##########

        all_generated_tokens_logprobs = []
        for completion_id in range(raw_request["num_return_sequences"]):
            generated_tokens_logprobs = []
            found_eos=False
            for i in range(len(sequences[completion_id]) - len(encoded_input.input_ids[0])):
                if(found_eos):
                    generated_tokens_logprobs.append(0)
                else:
                    logprobs = torch.nn.functional.log_softmax(scores[i][completion_id], dim=0)
                    # Get log probability of chosen token.
                    j = i + len(encoded_input.input_ids[0])
                    token_id=sequences[completion_id][j]
                    generated_tokens_logprobs.append(logprobs[token_id].item())
                    found_eos=(token_id==eos_token)

                    #this is just debugging
                    if(found_eos):
                        print("\n\n\n\n")
                        probs, indices= torch.topk(scores[i][completion_id], 5,largest=True)
                        with self.wrapped_tokenizer as tokenizer:
                            words=tokenizer.batch_decode(indices)
                            print("Highest probs for eos token:", list(zip(probs, words)))
            all_generated_tokens_logprobs.append(generated_tokens_logprobs)


        if not raw_request["echo_prompt"]:
            sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]


        ########## THEIR ATTEMPT ##########



        ########## MY ATTEMPT ##########
        # breakpoint()
        if raw_request["echo_prompt"]:
            raise Exception("I kind of broke echo prompt")

        # Compute logprobs of generated tokens for each completed sequence.
        all_generated_tokens_logprobs = []
        for completion_id in range(num_generated):
            generated_tokens_logprobs = []
            found_eos=False
            for i in range(len(sequences[completion_id])):
                if(found_eos):
                    generated_tokens_logprobs.append(0)
                else:
                    token_id=sequences[completion_id][i]
                    
                    logprobs = torch.nn.functional.log_softmax(scores[i][completion_id], dim=0)
                    generated_tokens_logprobs.append(logprobs[token_id].item())
                    found_eos=(token_id==eos_token)
                    if(found_eos):
                        print("\n\n\n\n")
                        probs, indices= torch.topk(scores[i][completion_id], 5,largest=True)
                        with self.wrapped_tokenizer as tokenizer:
                            words=tokenizer.batch_decode(indices)
                            print("Highest probs for eos token:", list(zip(probs, words)))
            all_generated_tokens_logprobs.append(generated_tokens_logprobs)

        ########## MY ATTEMPT ##########
        
        with self.wrapped_tokenizer as tokenizer:
            all_tokens = [[tokenizer.decode(token) for token in sequence_tokens] for sequence_tokens in sequences]
            all_decoded_text = tokenizer.batch_decode(sequences)
        raw_completions = []
        for decoded_text, tokens, generated_tokens_logprobs in zip(
            all_decoded_text, all_tokens, all_generated_tokens_logprobs
        ):
            # print("\n\n\n\n\n logprobs is", generated_tokens_logprobs)


            raw_completions.append(
                {
                    "text": decoded_text,
                    "tokens": tokens,
                    "logprobs": generated_tokens_logprobs,
                    "prompt_logprobs": prompt_tokens_logprobs,
                }
            )
        completions = raw_completions[:raw_request["num_return_sequences"]]
        return {"completions": completions, "input_length": len(encoded_input.input_ids[0]), "unscored_examples":raw_completions}


class HuggingFaceServerFactory:
    """A factory that creates and caches HuggingFaceServer objects."""

    _servers: Dict[str, HuggingFaceServer] = {}
    _servers_lock: Lock = Lock()

    @staticmethod
    def get_server(
        helm_model_name: str,
        pretrained_model_name_or_path: str,
        wrapped_tokenizer: WrappedPreTrainedTokenizer,
        **kwargs,
    ) -> Any:
        """
        Checks if the desired HuggingFaceModel is cached. Creates the HuggingFaceModel if it's not cached.
        Returns the HuggingFaceModel.
        """
        with HuggingFaceServerFactory._servers_lock:
            if helm_model_name not in HuggingFaceServerFactory._servers:
                with htrack_block(
                    f"Loading {pretrained_model_name_or_path} (kwargs={kwargs}) "
                    f"for HELM model {helm_model_name} with Hugging Face Transformers"
                ):
                    HuggingFaceServerFactory._servers[helm_model_name] = HuggingFaceServer(
                        pretrained_model_name_or_path, wrapped_tokenizer, **kwargs
                    )

        return HuggingFaceServerFactory._servers[helm_model_name]


TORCH_DTYPE_KEY = "torch_dtype"
TORCH_DTYPE_VALUE_PREFIX = "torch."


def _process_huggingface_client_kwargs(raw_kwargs: Dict[str, Any]):
    """Process the kwargs for HuggingFaceClient.

    The kwargs passed to HuggingFaceClient will eventually be passed to AutoModel.from_pretrained().
    Since the kwargs from HuggingFaceClient may be derived from configuration YAML,
    they may contain primitive types instead of the unserializable types that
    AutoModel.from_pretrained() expects (e.g. torch_dtype). This function converts values of
    primitive types to values of the unserializable types."""
    processed_kwargs = deepcopy(raw_kwargs)

    # Convert torch_dtype string value to actual dtypes
    # e.g. the string "torch.bfloat16" is converted to torch.bfloat16
    torch_dtype = processed_kwargs.get(TORCH_DTYPE_KEY)
    if torch_dtype and isinstance(torch_dtype, str):
        if torch_dtype.startswith(TORCH_DTYPE_VALUE_PREFIX):
            processed_kwargs[TORCH_DTYPE_KEY] = getattr(torch, torch_dtype[len(TORCH_DTYPE_VALUE_PREFIX) :])

    return processed_kwargs


class HuggingFaceClient(CachingClient):
    def __init__(
        self,
        cache_config: CacheConfig,
        tokenizer: Tokenizer,
        pretrained_model_name_or_path: Optional[str] = None,
        end_of_text_token: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(cache_config=cache_config)
        self._pretrained_model_name_or_path = pretrained_model_name_or_path
        if not isinstance(tokenizer, HuggingFaceTokenizer):
            raise ValueError(
                f"Tokenizer for Hugging Face model {pretrained_model_name_or_path} must be a HuggingFaceTokenizer, "
                "but instead it is {tokenizer}"
            )
        self._wrapped_tokenizer: WrappedPreTrainedTokenizer = tokenizer.get_wrapped_tokenizer()
        self._tokenizer = tokenizer
        self._kwargs = _process_huggingface_client_kwargs(kwargs)
        self._end_of_text_token = end_of_text_token
        self._lock= Lock()
        self._output_file="completions.txt"


    def clean_completions(self, response, request, completions_to_clean, should_truncate_sequence=True):

        completions = []
        for raw_completion in completions_to_clean:
            sequence_logprob: float = 0
            tokens: List[Token] = []

            # if request.echo_prompt:
            #     # Add prompt to list of generated tokens.
            #     generated_tokens = raw_completion["tokens"][response["input_length"] :]
            #     if raw_completion.get("prompt_logprobs"):
            #         for token_text, logprob in zip(
            #             raw_completion["tokens"][: response["input_length"]],
            #             raw_completion["prompt_logprobs"][: response["input_length"]],
            #         ):
            #             tokens.append(Token(text=token_text, logprob=logprob))
            #             sequence_logprob += logprob
            #     else:
            #         for token_text in raw_completion["tokens"][: response["input_length"]]:
            #             tokens.append(Token(text=token_text, logprob=0.0))

            # else:
            generated_tokens = raw_completion["tokens"]

            # Compute logprob for the entire sequence.
            for token_text, logprob in zip(generated_tokens, raw_completion["logprobs"]):
                tokens.append(Token(text=token_text, logprob=logprob))
                sequence_logprob += logprob

            completion = GeneratedOutput(text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens)
            if(should_truncate_sequence):
                completion = truncate_sequence(completion, request, end_of_text_token=self._end_of_text_token)
            completions.append(completion)
        return completions

    def make_request(self, request: Request) -> RequestResult:

        
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT
        

        # print("\n\n\n\n\n LUKE: Original Request is ",request)

        raw_request: HuggingFaceRequest = {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "temperature": 1e-7 if request.temperature == 0 else request.temperature,
            "num_beams": request.num_beams,
            "generated_output_file": request.generated_output_file,
            "num_return_sequences": request.num_completions,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
            "echo_prompt": request.echo_prompt,
            "top_k_per_token": request.top_k_per_token,
            "stop_sequences": request.stop_sequences
        }
        
        pretrained_model_name_or_path = (
            self._pretrained_model_name_or_path if self._pretrained_model_name_or_path else request.model
        )
        huggingface_model: HuggingFaceServer = HuggingFaceServerFactory.get_server(
            helm_model_name=request.model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            wrapped_tokenizer=self._wrapped_tokenizer,
            **self._kwargs,
        )
        def do_it() -> Dict[str, Any]:
            return huggingface_model.serve_request(raw_request)
        cache_key = CachingClient.make_cache_key(raw_request, request)
        response, cached = self.cache.get(cache_key, wrap_request_time(do_it))

        # try:
        #     cache_key = CachingClient.make_cache_key(raw_request, request)
        #     response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        # except Exception as e:  # Do something if error is encountered.
        #     error: str = f"HuggingFace error: {e}"
        #     return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions = self.clean_completions(response, request,response["completions"],should_truncate_sequence=True)
        unscored_examples = self.clean_completions(response, request, response["unscored_examples"],should_truncate_sequence=False)
        # for completion in unscored_examples:
            #completion.tokens=None

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            unscored_examples=unscored_examples,
            embedding=[],
        )
