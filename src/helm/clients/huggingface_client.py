from copy import deepcopy
import torch
from transformers import AutoModelForCausalLM
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
    EosTokenCriteria
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
    BeamParams
)
from helm.tokenizers.tokenizer import Tokenizer
from helm.clients.client import CachingClient, truncate_sequence
from helm.tokenizers.huggingface_tokenizer import HuggingFaceTokenizer, WrappedPreTrainedTokenizer
from threading import Lock
import json
from pprint import pprint
import sys

# class StopAtSpecificTokenCriteria(StoppingCriteria):
#     def __init__(self, stop_sequence: List[int]):
#         super().__init__()
#         self.stop_sequence = stop_sequence

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         # Create a tensor from the stop_sequence
#         stop_sequence_tensor = torch.tensor(self.stop_sequence, device=input_ids.device, dtype=input_ids.dtype)

#         # Check if the current sequence ends with the stop_sequence
#         current_sequence = input_ids[:, -len(self.stop_sequence) :]
#         return bool(torch.all(current_sequence == stop_sequence_tensor).item())

class StopOnStrings(StoppingCriteria):
    def __init__(self, stop_strings, tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.stream = ""
        self.prev_generated= ""

    def reset(self):
        self.stream = ""

    def __call__(self, input_ids, scores, **kwargs):

        # print("input_ids is ",input_ids.shape)
        generated = self.tokenizer.decode(input_ids[0][-1], skip_special_tokens=True)
        for stop_string in self.stop_strings:
            if stop_string in generated:
                return True
        return False
    
    def __call__(self, input_ids, scores, **kwargs):
        generated = self.tokenizer.decode(input_ids[0][-1], skip_special_tokens=True)
        self.stream += generated
        for stop_string in self.stop_strings:
            if stop_string in self.stream[:-1]:
                # print(f"stop_string is {stop_string}, stream is{self.stream}")
                # print("hello")
                return True
        return False


def exact_mode_algo(model, x:str,bos:int, eos:int, wrapped_tokenizer):
    def get_next_log_probs(x, y, model):
        with torch.no_grad():
            outputs = model(input_ids=x, decoder_input_ids=y)
            logits = outputs.logits
        
        next_token_logits = logits[:, -1, :]
        return torch.nn.functional.log_softmax(next_token_logits, dim=-1)[0]
    def DFS(  x:str,  y:str, p:float, gamma:float,model, eos=1):
        if(y[0,-1]==eos):
            return (y,p)
        best_y=None
        
        #exclude the pad token
        log_probs=get_next_log_probs(x, y, model)
        for idx, log_prob in enumerate(log_probs):
            if(idx>0):
                newP = p + log_prob 
                if newP >= gamma:
                    # print("newP ",newP," p ",p," log_prob ",log_prob)
                    appended_y=torch.concat((y, torch.tensor([[idx]], dtype=int)), axis=1)
                    new_y, new_gamma = DFS(  x,  appended_y, newP, gamma, model, eos)
                    if new_gamma > gamma:
                        best_y=new_y
                        gamma=new_gamma
        return best_y, gamma
    def get_decode_log_prob(x,output, model, wrapped_tokenizer):
        score=0
        for idx in range(1, output.size()[1]):
            pred=output[0, idx]
            y=output[:, :idx]
            log_p= get_next_log_probs(x,y , model)[pred]
            score+=log_p    
        with wrapped_tokenizer as tokenizer:
            print("gen_outputs ",score," : ",tokenizer.decode(y[0], skip_special_tokens=True))
        return score
    
    y = bos*torch.ones((1,1), dtype=int)

    ended_y=torch.concat((y, torch.tensor([[eos]], dtype=int)), axis=1)
    start_gamma=get_next_log_probs(x, ended_y, model)[eos]
    best_y, gamma = DFS(x, y,0, start_gamma, model, eos)
    print(f"best_y is {best_y}")
    print(f"Gamma is ",gamma)
    get_decode_log_prob(x,best_y, model, wrapped_tokenizer)
    return best_y, gamma

class HuggingFaceRequest(TypedDict):
    """Data passed between make_request and serve_request. Used as the cache key."""

    engine: str
    prompt: str
    temperature: float
    # num_beams:int
    beam_params:dict
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
        end_of_text_token:str,
        **kwargs,
    ):
        self._lock= Lock()
        self.stop_sequence_dict={}
        self.device: Optional[str]
        # print(f"\n\n\n\n kwargs is {kwargs}")
        if torch.cuda.is_available():
            kwargs["device_map"]="auto"
        print(f"kwargs is {kwargs}")
        if "device_map" in kwargs:
            if "device" in kwargs:
                raise ValueError("At most one of one of `device` and `device_map` may be specified.")
            try:
                import accelerate  # noqa: F401
                
            except ModuleNotFoundError as e:
                print("accelerate not installed!!!")
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

        self._end_of_text_token=end_of_text_token

        with wrapped_tokenizer as tokenizer:
            self.eos=tokenizer.eos_token
            self.bos=tokenizer.bos_token
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
        print("Serving request", flush=True)
        eos_token_string=None
        stopping_criteria: Optional[StoppingCriteriaList] = None
        optional_args = {}
        prompt=raw_request["prompt"]

        prompt_tokens_logprobs = []
        stop_strings=[self.eos]
        all_generated_tokens_logprobs = []
        with self.wrapped_tokenizer as tokenizer:
            encoded_input = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(
                0 if self.device is None else self.device
            )
            # if len(raw_request["stop_sequences"]) > 0:
            #     stopping_criteria = StoppingCriteriaList()
            #     stop_strings=stop_strings+raw_request["stop_sequences"]
            #     # breakpoint()
            #     stopping_criteria.append(StopOnStrings(stop_strings=stop_strings,tokenizer=tokenizer))


                # stopping_criteria.append(StopStringCriteria(tokenizer, [",","<|endoftext|>"]))
                
                # stopping_criteria.append(EosTokenCriteria(eos_token_id=13))
                

        # if len(raw_request["stop_sequences"]) > 0:
            # with self.wrapped_tokenizer as tokenizer:
            #     stop_sequence_ids = tokenizer(
            #         raw_request["stop_sequences"], return_token_type_ids=False, add_special_tokens=False
            #     )
            # if len(stop_sequence_ids.input_ids) == 1 and len(stop_sequence_ids.input_ids[0]) == 1:
            #     optional_args["eos_token_id"] = stop_sequence_ids.input_ids[0][0]
            # else:
            #     stopping_criteria = StoppingCriteriaList()
            #     for stop_sequence_input_ids in stop_sequence_ids.input_ids:
                    
            #         stopping_criteria.append(StopOnStrings(stop_sequence=stop_sequence_input_ids))


        # Check if we need to compute the perplexity of the prompt (#1497)
        compute_logprobs_only = (
            raw_request["max_new_tokens"] == 0
            and raw_request["num_return_sequences"] == 1
            and raw_request["echo_prompt"]
        )


        # num_beams= int(raw_request["num_beams"]) if ("num_beams" in raw_request.keys() ) else None
        num_beams=raw_request["beam_params"].num_beams
        raw_num_return_sequences=raw_request["beam_params"].num_return_sequences
        top_p=raw_request["beam_params"].top_p
        top_k=raw_request["beam_params"].top_k
        if top_k<1:
            top_k=sys.maxsize
        # raw_num_return_sequences=int(raw_request["num_return_sequences"])
        
        num_generated= raw_num_return_sequences if num_beams is None else max(raw_num_return_sequences, num_beams)
        assert(raw_request["top_p"]==1)

        # Use HuggingFace's `generate` method.
        if compute_logprobs_only:
            with torch.no_grad():
                output = self.model(encoded_input["input_ids"])
            sequences = encoded_input["input_ids"]
            scores = output.logits
        else:

            #beam search
            if(num_beams >1):
                # should be this?
                # with torch.no_grad():
                    # outputs = self.model.generate(
                    #     **encoded_input,
                    #     num_beams = num_beams,
                    #     num_return_sequences=num_generated,
                    #     max_new_tokens=raw_request["max_new_tokens"],
                    #     #changed this
                    #     do_sample=False,
                    #     return_dict_in_generate=True,
                    #     output_scores=True,
                    #     output_logits=True,
                    #     length_penalty=0,
                    #     **optional_args,
                    #     stopping_criteria=stopping_criteria, 
                    #     early_stopping=False,
                    #     top_k = 0

                    # )
                with torch.no_grad():
                    output = self.model.generate(**encoded_input, 
                            max_new_tokens=raw_request["max_new_tokens"], 
                            num_beams=num_beams,
                            num_return_sequences=num_generated,
                            do_sample=False,
                            return_dict_in_generate=True,
                            output_scores=True,
                            output_logits=True,
                            length_penalty=0,
                             **optional_args,
                            early_stopping="never", 
                            stopping_criteria=stopping_criteria, 
                            # num_beam_groups=num_beams,
                            # diversity_penalty=1.0,
                            )
                # with self.wrapped_tokenizer as tokenizer:
                    sequences = output.sequences
                    scores = output.scores
                    logits=output.logits
                    transition_scores = self.model.compute_transition_scores(output.sequences, output.scores, output.beam_indices, normalize_logits=True)
                    # Compute logprobs of generated tokens for each completed sequence.
                    for completion_id in range(num_generated):
                        generated_tokens_logprobs=[]
                        generated_sequence=sequences[completion_id, len(encoded_input.input_ids[0]):]
                        sentence_length=min( len(generated_sequence), len(logits))
                        for i in range(sentence_length): 
                            token_logprob=transition_scores[completion_id][i].item()
                            generated_tokens_logprobs.append(token_logprob)
                        all_generated_tokens_logprobs.append(generated_tokens_logprobs)
            
            #exact mode
            elif num_beams==-1:
                exact_mode_algo(model=self.model, x=encoded_input,bos=self.bos, eos=self.eos, wrapped_tokenizer=self.wrapped_tokenizer)
            
            #helm default
            elif num_beams is None or num_beams==0: 

                #Defaults
                with torch.no_grad():
                    output = self.model.generate(
                        **encoded_input,
                        temperature=raw_request["temperature"],
                        num_return_sequences=num_generated,
                        max_new_tokens=raw_request["max_new_tokens"],
                        top_p=raw_request["top_p"],
                        do_sample=True,
                        return_dict_in_generate=True,
                        output_scores=True,
                        output_logits=True,
                        **optional_args,
                        stopping_criteria=stopping_criteria,
                    )

                sequences = output.sequences
                logits = output.logits
                
                for completion_id in range(num_generated):
                    generated_tokens_logprobs = []
                    for i in range(len(sequences[completion_id]) - len(encoded_input.input_ids[0])):
                        logprobs = torch.nn.functional.log_softmax(logits[i][completion_id], dim=0)
                        # Get log probability of chosen token.
                        j = i + len(encoded_input.input_ids[0])
                        generated_tokens_logprobs.append(logprobs[sequences[completion_id][j]].item())
                    all_generated_tokens_logprobs.append(generated_tokens_logprobs)
            
            #non-beam search tests
            #default for test_run_all.ksh
            elif num_beams==1:
                #unbiased sampling
                output = self.model.generate(
                    **encoded_input,
                    num_return_sequences=num_generated,
                    max_new_tokens=raw_request["max_new_tokens"],
                    length_penalty=1,
                    top_p=top_p,
                    top_k=top_k,
                    # top_k=sys.maxsize,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_logits=True,
                    **optional_args,
                    stopping_criteria=stopping_criteria,
                )
                sequences = output.sequences
                logits = output.logits
                
                for completion_id in range(num_generated):
                    generated_tokens_logprobs = []
                    for i in range(len(sequences[completion_id]) - len(encoded_input.input_ids[0])):
                        logprobs = torch.nn.functional.log_softmax(logits[i][completion_id], dim=0)
                        # Get log probability of chosen token.
                        j = i + len(encoded_input.input_ids[0])
                        generated_tokens_logprobs.append(logprobs[sequences[completion_id][j]].item())
                    all_generated_tokens_logprobs.append(generated_tokens_logprobs)
            else:
                raise Exception(f"Weird number of num_beams {num_beams}")
        
        # for completion_id in range(num_generated):
        #     generated_sequence=sequences[completion_id, len(encoded_input.input_ids[0]):]
        #     # print("sequence is ", tokenizer.batch_decode(generated_sequence, skip_special_tokens=True))
        #     generated_tokens_logprobs = []
        #     sentence_length=min( len(generated_sequence), len(transition_scores))
        #     for i in range(sentence_length): 
        #         cur_token=generated_sequence[i]
        #         token_logprob=transition_scores[completion_id][i].item()
        #         # breakpoint()
        #         # logprobs = torch.nn.functional.log_softmax(logits[i][completion_id], dim=-1)
        #         # token_logprob=logprobs[cur_token].item()
        #         # token_score=output.scores[i][completion_id][cur_token].item()
        #         # assert token_score==token_logprob  
        #         # print(f"cur_token is {tokenizer.decode(cur_token)} \tscore: {token_score} \ttoken_logprob {token_logprob}")     
        #         generated_tokens_logprobs.append(token_logprob)      
        #     all_generated_tokens_logprobs.append(generated_tokens_logprobs)

        # Remove prompt from the start of each sequence if echo_prompt is False.
        if not raw_request["echo_prompt"]:
            sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]
        
        with self.wrapped_tokenizer as tokenizer:
            all_tokens = [[tokenizer.decode(token) for token in sequence_tokens] for sequence_tokens in sequences]
            all_decoded_text = tokenizer.batch_decode(sequences)
        raw_completions = []

        # if(num_beams==1):
        #     line_char="----------------------------------------------------------------"
        #     print("Next instance")
        #     print("\n"+line_char)
        #     print(line_char)
        #     smaller_prompt=raw_request["prompt"]
        #     while("\n\n\n" in smaller_prompt):
        #         smaller_prompt=smaller_prompt.replace("\n\n\n","\n\n")
        #     print(f'Prompt:\n {smaller_prompt}')
        #     print(line_char)
        #     all_decoded_text = tokenizer.batch_decode(sequences)
        #     print(f"Decoded:\n {all_decoded_text[0]}")
        #     print(f"{line_char}")
        #     print("tokens:")
        #     # for tokens,generated_tokens_logprobs  in zip(all_tokens,all_generated_tokens_logprobs):
        #     #     for token, logprob in zip(tokens,generated_tokens_logprobs):
        #     #         print(f"{token}: {logprob}")
        #     print(f"{line_char}")
        
        for decoded_text, tokens, generated_tokens_logprobs, sequence in zip(
            all_decoded_text, all_tokens, all_generated_tokens_logprobs, sequences
        ):
            raw_completions.append(
                {
                    "text": decoded_text,
                    "tokens": tokens,
                    "sequence":sequence,
                    "logprobs": generated_tokens_logprobs,
                    "prompt_logprobs": prompt_tokens_logprobs
                }
            )
        


        raw_completions.sort(key=lambda x:sum(x["logprobs"]),reverse=True)
        completions = raw_completions[:num_generated]


        # with open('debug.txt', 'w') as f:
        #     txt=completions[0].txt.split("<|endoftext|>")[0]
        #     txt=txt.split("<|endoftext|>")[0]
        #     f.write(f"Prompt is {raw_request['prompt']}")
        #     f.write(f"Completion is {}")
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
        end_of_text_token: str,
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
                        pretrained_model_name_or_path, wrapped_tokenizer, end_of_text_token, **kwargs
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
        
        if self._end_of_text_token==None:
            with self._wrapped_tokenizer as mytokenizer:
                self._end_of_text_token=mytokenizer.special_tokens_map['eos_token']
        self._lock= Lock()
        self._output_file="completions.txt"
        # print("\nClient end_of_text_token",self._end_of_text_token)


    def clean_completions(self, response, request, completions_to_clean, should_truncate_sequence=True):

        #completion: calculate up until \n
        #tokens: calculate up until eos
        #token probability: calculate up until eos
        
        completions = []
        for idx, raw_completion in enumerate(completions_to_clean):
            sequence_logprob: float = 0
            tokens: List[Token] = []
            # print("raw_completion is ",raw_completion)
            # print("\n\nText is ",raw_completion["text"])
            # print("raw_completion.keys is ", raw_completion.keys())
            generated_tokens = raw_completion["tokens"]
            sequence=raw_completion["sequence"]

            end_of_text_token=raw_completion["end_of_text_token"] if "end_of_text_token" in raw_completion.keys() else self._end_of_text_token

            # Compute logprob for the entire sequence.
            for token_text, logprob, token_id in zip(generated_tokens, raw_completion["logprobs"], sequence):
                tokens.append(Token(text=token_text, logprob=logprob, token_id=token_id.item()))
                sequence_logprob += logprob

                if(token_text==end_of_text_token):
                    break
            completion = GeneratedOutput(text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens, example_id=idx)
            if(should_truncate_sequence):
                completion = truncate_sequence(completion, request, end_of_text_token=end_of_text_token)
            completions.append(completion)



            # 
            # # if(completion.full_text):
            # #     print("full text: ",completion.full_text)

            # # "<|endoftext|>"
            # # "."=="13"
            # # "<|endoftext|>"=50256
            # # end_token_found=False
            # # for token in completion.tokens:
            # #     if token.token_id==13:
            # #         end_token_found=True
            # #     if end_token_found and token.text != self._end_of_text_token:
            # #         print("error found ")
            # #         print("\n\n\n\n\n\n\n\n\n\n\n\n\n",completion.full_text)
                    
        return completions

    def make_request(self, request: Request) -> RequestResult:

        
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT
        

        prompt=request.prompt.replace("<|helm_eot_id|>", self._end_of_text_token)
        # print("-------------\n\n\n\n prompt is ",prompt )
        # beam_params = 
        
        raw_request: HuggingFaceRequest = {
            "engine": request.model_engine,
            "prompt": prompt,
            "temperature": 1e-7 if request.temperature == 0 else request.temperature,
            # "num_beams": request.num_beams,
            "beam_params": request.beam_params,
            "generated_output_file": request.generated_output_file,
            "num_return_sequences": request.num_completions,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
            "echo_prompt": request.echo_prompt,
            "top_k_per_token": request.top_k_per_token,
            "stop_sequences": request.stop_sequences
        }

        if request.beam_params.num_beams > 1:
            assert request.num_completions == 1

        pretrained_model_name_or_path = (
            self._pretrained_model_name_or_path if self._pretrained_model_name_or_path else request.model
        )
        huggingface_model: HuggingFaceServer = HuggingFaceServerFactory.get_server(
            helm_model_name=request.model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            wrapped_tokenizer=self._wrapped_tokenizer,
            end_of_text_token=self._end_of_text_token,
            **self._kwargs,
        )

        expose_error=True
        if(expose_error):
            def do_it() -> Dict[str, Any]:
                return huggingface_model.serve_request(raw_request)
            cache_key = CachingClient.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        else:
            try:
                def do_it() -> Dict[str, Any]:
                    return huggingface_model.serve_request(raw_request)
                cache_key = CachingClient.make_cache_key(raw_request, request)
                response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            except Exception as e:  # Do something if error is encountered.
                error: str = f"HuggingFace error: {e}"
                return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions = self.clean_completions(response, request,response["completions"],should_truncate_sequence=True)
        unscored_examples = self.clean_completions(response, request, response["unscored_examples"],should_truncate_sequence=True)


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
            full_prompt=raw_request["prompt"]
        )
