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
import gc
import time

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
import numpy as np
from pynvml import NVMLError

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


import torch
import time
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    for retry_num in range(max_retries):
        info = nvmlDeviceGetMemoryInfo(handle)
        free = float(info.free)/(1024 * 1024 * 1024)

        if info.free >= min_memory_available:
            print(f"Recovered cuda. Freed up {free} GB of GPU")
            break
        print(f"Requested: {min_memory_available/(1024 * 1024 * 1024)}GB. Available: {free} GB. Not enough Gpu GB Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    else:
        print("\n\n\n\n-----------------------\nPrint all tensors", flush=True)
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size())
            except:
                pass
        print(torch.cuda.memory_summary(device=None, abbreviated=False), flush=True)
        raise RuntimeError(f"Failed to acquire {min_memory_available/(1024 * 1024 * 1024)} GB of free GPU memory after {max_retries} retries.")


def pad_to_dim(m, correct_sizes, axes, num_dim, cat_axis, pad_value):
    
    pad_tuple = [0]*(2*num_dim)

    for axis in axes:
        if axis != cat_axis:
            diff = correct_sizes[axis]-m.size(axis) 
            if  diff != 0:
                pad_tuple[2*((num_dim-1)-axis)+1] = diff
    return torch.nn.functional.pad(input=m, pad=pad_tuple, value=pad_value)
    

def match_sizes(m1, m2,cat_axis, pad_value):
    #if they're the same size
    s1=list(m1.size())
    s2=list(m2.size())
    s1.pop(cat_axis)
    s2.pop(cat_axis)
    if s1 == s2:
        return m1, m2

    num_dim=len(m1.size())
    axes=list(range(num_dim))
    correct_sizes= [max(m1.size(axis),m2.size(axis)) for axis in axes]
    m1=pad_to_dim(m1, correct_sizes, axes, num_dim, cat_axis, pad_value)
    m2=pad_to_dim(m2, correct_sizes, axes, num_dim, cat_axis, pad_value)
    return m1, m2


def safe_append_tensor(tensor_agg, batch_tensor, cat_axis, pad_value):
    if tensor_agg is None:
        return batch_tensor
    
    tensor_agg, batch_tensor=match_sizes(tensor_agg, batch_tensor,cat_axis, pad_value)
    return  torch.cat((tensor_agg,batch_tensor), axis=cat_axis)

def safe_append_list(list_agg, new_list):
    return new_list if list_agg is None else list_agg+new_list


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


def exact_mode_algo(model, encoded_input, eos:int):
    def get_next_log_probs(y, model):
        with torch.no_grad():
            outputs = model(input_ids=y)
            logits = outputs.logits
        
        next_token_logits = logits[:, -1, :]
        return torch.nn.functional.log_softmax(next_token_logits, dim=-1)[0]

    #y is the prompt with additions, p is the probability of y, gamma is the current best probability, eos is the eos string
    def DFS(  y:str, p:float, gamma:float,model, eos=1, depth=0, max_depth=-1):
        #If we reached max depth without finishing
        if(depth>max_depth):
            return (y,gamma*2)
        
        #if y is finished, return the node
        if(y[0,-1]==eos):
            print(f"p is {p}, y is {y}, gamma is {gamma}", flush=True)
            return (y,p)
        
        #exclude the pad token
        log_probs=get_next_log_probs(y, model)

        arange=torch.arange(len(log_probs)).to(log_probs.device)
        best_y=y
        for idx, log_prob in enumerate(log_probs):
            newP = p + log_prob 
            #if we're doing better than the best one so far
            if newP > gamma:
                #do a DFS
                appended_y=torch.concat((y, arange[idx].reshape(1,1)), axis=1)
                new_y, new_gamma = DFS( y=appended_y, p=newP, gamma=gamma, model=model, eos=eos, depth=depth+1, max_depth=max_depth)
                if new_gamma > gamma:
                    best_y=new_y
                    gamma=new_gamma

        return best_y, gamma

    y=encoded_input.input_ids
    ended_y=torch.concat((y, eos), axis=1)
    start_gamma=get_next_log_probs(y=ended_y, model=model)[eos]
    best_y, gamma = DFS(y=y, gamma=start_gamma,p=0, model= model, eos=eos, depth=0, max_depth=100)
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
    model=None
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        wrapped_tokenizer: WrappedPreTrainedTokenizer,
        end_of_text_token:str,
        **kwargs,
    ):

        torch.set_float32_matmul_precision('medium')
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
            self.eos_id=  tokenizer(tokenizer.eos_token, return_tensors="pt", return_token_type_ids=False).input_ids.flatten()[0].item()
            self.eos_id_tensor = torch.tensor([self.eos_id]).reshape(1,1).to(0 if self.device is None else self.device)
            
        # Security issue: currently we trust remote code by default.
        # We retain this temporarily to maintain reverse compatibility.
        # TODO: Delete if-else and don't set trust_remote_code=True
        if "trust_remote_code" not in kwargs:
            kwargs["trust_remote_code"] = True

        self.pretrained_model_name_or_path=pretrained_model_name_or_path
        self.model_kwargs=kwargs
        with htrack_block(f"Loading Hugging Face model {pretrained_model_name_or_path}"):
            # WARNING this may fail if your GPU does not have enough memory
            self.set_model()


        self.wrapped_tokenizer = wrapped_tokenizer
        self.batch_size=None

        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            info = nvmlDeviceGetMemoryInfo(handle)
            self.initial_free = float(info.free)/(1024 * 1024 * 1024)
        except NVMLError as err:
            self.initial_free=0


        # print(f"intitial_free is {self.initial_free }", flush=True)
    def recover_from_oom(self):
        self.lower_batch_size()
        
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        #wait for memory to clear up
        available_percent=0.8
        min_memory_available = available_percent* self.initial_free  * 1024 * 1024 * 1024  # 60GB is max. Get 80% of it

        wait_until_enough_gpu_memory(min_memory_available)


    def set_model(self):
        if self.model is None:
            if self.device is None:
                    # kwargs contains device_map=auto
                    # Do not call to() because accelerate will take care of model device placement.
                self.model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name_or_path, **self.model_kwargs)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name_or_path, **self.model_kwargs).to(
                    self.device
                )
        
            print(f"Model dtype: {next(self.model.parameters()).dtype}")                

    def decode_text(self, sequences, input_len, echo_prompt=False):
        
        if not echo_prompt:
            sequences = [sequence[input_len:] for sequence in sequences]

        with self.wrapped_tokenizer as tokenizer:
            all_tokens = [[tokenizer.decode(token) for token in sequence_tokens] for sequence_tokens in sequences]
            all_decoded_text = tokenizer.batch_decode(sequences)
        return all_tokens, all_decoded_text
                
    def lower_batch_size(self):
        def get_next_smaller(num):
            options=[100, 75, 50, 30, 20, 15, 12, 10, 9, 8,7,6,5,4,3,2,1]
            # options=[100,1]
            if num is None:
                return options[0]
            for option in options:
                if(option< num):
                    return option
            raise Exception(f"Exception: Could not find smaller number than {num}")
        self.batch_size=get_next_smaller(self.batch_size)


    def serve_request(self, raw_request: HuggingFaceRequest) -> Dict:
        self.set_model()
        # print(f"Serving request.", flush=True)
        eos_token_string=None
        logits=None
        sequences=None
        all_decoded_text=None
        all_tokens=None



        #don't have None batch size
        if  self.batch_size is None:
            self.batch_size=1000 if  (raw_request["beam_params"].batch_size  is None) else raw_request["beam_params"].batch_size

        # #fake exception
        # if self.batch_size>5:
        #     print(f"Failed: batch_size is {self.batch_size}")
        #     raise Exception("torch.OutOfMemoryError")
        # else:
        #     print(f"Success: batch_size is {self.batch_size}")


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
        temperature=raw_request["beam_params"].temperature
        length_penalty=raw_request["beam_params"].length_penalty
        exact_mode=raw_request["beam_params"].exact_mode
        
        batch_size=self.batch_size

        
        
        
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


            #exact mode
            if exact_mode==True:
                # best_y, gamma= exact_mode_algo(model=self.model, x=encoded_input,bos=self.bos, eos=self.eos, wrapped_tokenizer=self.wrapped_tokenizer)
                best_y, gamma=exact_mode_algo(self.model, encoded_input, self.eos_id_tensor)

                # print(f"self.device is {self.device}")    

                input_ids=encoded_input.input_ids
                best_y = best_y.to(input_ids.device)
                # print(f"encoded_input cuda is {input_ids.is_cuda}")
                # print(f"best_y cuda is {best_y.is_cuda}")
                # print(f"input type {type(input_ids)}")
                # print(f"best_y type {type(best_y)}")

                full_sentence=torch.concat((input_ids, best_y), axis=1)
                with torch.no_grad():
                    # #this is a test
                    # with torch.no_grad():
                    #     output0 = self.model(encoded_input["input_ids"])
                    #     sequences0 = encoded_input["input_ids"]
                    #     scores0 = output0.logits                    
                    output = self.model(full_sentence)
                    sequences = full_sentence
                    logits = output.logits

                completion_id=0
                generated_tokens_logprobs = []
                for i in range(len(sequences[completion_id]) - len(encoded_input.input_ids[0])):
                    logprobs = torch.nn.functional.log_softmax(logits[i][completion_id], dim=0)
                    # Get log probability of chosen token.
                    j = i + len(encoded_input.input_ids[0])
                    generated_tokens_logprobs.append(logprobs[sequences[completion_id][j]].item())
                all_generated_tokens_logprobs.append(generated_tokens_logprobs)

            elif(num_beams >1):
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
                try: 
                    print(f"Serving request. Batch size: {self.batch_size}", flush=True)
                    batch_output=None
                    batch_sequences=None
                    batch_logits=None
                    batch_tokens=None
                    batch_decoded_text=None
                    all_tokens=None
                    all_decoded_text=None
                    sequences=None

                    batch_size=min(self.batch_size, batch_size)
                    batch_size = num_generated if batch_size == 0 else batch_size
                    num_left=num_generated
                    while(num_left>0):
                        new_batch=min(num_left, batch_size, self.batch_size)
                        num_left -= new_batch
                    # for i in range(int(num_generated / batch_size)):
                        with torch.no_grad():
                            batch_output = self.model.generate(
                                **encoded_input,
                                num_return_sequences=new_batch,
                                max_new_tokens=raw_request["max_new_tokens"],

                                # length_penalty=length_penalty,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                do_sample=True,

                                return_dict_in_generate=True,
                                output_scores=True,
                                output_logits=True,
                                **optional_args,
                                stopping_criteria=stopping_criteria,
                            )
                        #generate
                        batch_sequences = batch_output.sequences
                        batch_logits=batch_output.logits

                        # batch_logits = torch.stack(list(batch_output.logits), dim=0)
                        # print(f"len is {len(all_generated_tokens_logprobs)}")
                        for completion_id in range(new_batch):
                            generated_tokens_logprobs = []
                            for i in range(len(batch_sequences[completion_id]) - len(encoded_input.input_ids[0])):
                                logprobs = torch.nn.functional.log_softmax(batch_logits[i][completion_id], dim=0)
                                # Get log probability of chosen token.
                                j = i + len(encoded_input.input_ids[0])
                                generated_tokens_logprobs.append(logprobs[batch_sequences[completion_id][j]].item())
                            all_generated_tokens_logprobs.append(generated_tokens_logprobs)
                            
                        #batch_tokens is a list of outputs. Each output is list of word-strings
                        #batch_decoded_text is a list of outputs. Each output is strings.
                        
                        batch_tokens, batch_decoded_text = self.decode_text(sequences=batch_sequences, input_len=len(encoded_input.input_ids[0]),echo_prompt=raw_request["echo_prompt"])
                        all_tokens= safe_append_list(all_tokens,batch_tokens)
                        all_decoded_text =safe_append_list(all_decoded_text,batch_decoded_text)

                        sequences = safe_append_list(sequences, list(batch_sequences.detach().cpu())  )

                        # print(sequences[0].is_cuda)
                        # sequences = safe_append_tensor(sequences, batch_sequences.detach().cpu(), 0, pad_value=self.eos_id)
                        # logits = safe_append_tensor(logits, batch_logits.detach().cpu(), 1, pad_value=-1)
                        successful=True
                except Exception as e: 
                    is_cuda_memory_error= ('CUDA out of memory. Tried to allocate' in str(e))
                    if is_cuda_memory_error:

                        
                        print("Deleting everything", flush=True)
                        batch_output=None                   
                        batch_sequences=None
                        batch_logits=None
                        batch_tokens=None
                        batch_decoded_text=None
                        all_tokens=None
                        all_decoded_text=None
                        sequences=None
                        self.model=None
                        encoded_input=None
                        all_generated_tokens_logprobs=None
                        generated_tokens_logprobs=None                        
                        return None
                    else:
                        raise e
            else:
                raise Exception(f"Weird number of num_beams {num_beams}")

        # Remove prompt from the start of each sequence if echo_prompt is False.        
        if all_tokens is None:
            all_tokens, all_decoded_text = self.decode_text(sequences=sequences, input_len=len(encoded_input.input_ids[0]),echo_prompt=raw_request["echo_prompt"])

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

        def do_it() -> Dict[str, Any]:
            num_attempts=20
            for i in range(num_attempts):
                return_value=huggingface_model.serve_request(raw_request)
                if return_value is None:
                    huggingface_model.recover_from_oom()
                    print(f"Attempt {i} unsucessful. Retrying")
                else:
                    return return_value
        expose_error=True
        if(expose_error):
            cache_key = CachingClient.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        else:
            try:
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
