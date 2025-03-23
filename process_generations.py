from src.helm.benchmark.runner import GeneratedOutputExamples
from transformers import AutoModelForCausalLM
from helm.tokenizers.huggingface_tokenizer import HuggingFaceTokenizer
from helm.common.request import (GeneratedOutput)


import json
import torch
json_file="benchmark_output/runs/run_all_evalnum_2_wmt_14_language_pair_de_en_model_distilbert_distilgpt2_follow_format_instructions_instruct_num_beams_2/wmt_14:language_pair=de-en,model=distilbert_distilgpt2/instance_generations.json"








#get tokens
kwargs={}
kwargs['trust_remote_code']=True
pretrained_model_name_or_path="distilbert/distilgpt2"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
wrapped_tokenizer = HuggingFaceTokenizer.create_tokenizer(pretrained_model_name_or_path)


prompt=""""Translate the following sentences from German to English.
German: Der Konferenz- und Tagungsbereich besteht aus fünf modern ausgestatteten Räumen für 5 – 30 Personen.
English: The conference area consists of five modern rooms suitable for 5 - 30 persons. We are your reliable partner for conferences, family gatherings, balls, receptions and catering.

German: Er riet den Eltern eines Jungen, dessen Penis bei einer verpfuschten Beschneidung abgetrennt worden war, das Kind ganz zu kastrieren und auch seine Hoden zu entfernen und ihn dann als Mädchen großzuziehen.
English: The conference area consists of five modern rooms suitable for 5 - 30 persons. We are your reliable partner for conferences, family gatherings, balls, receptions and catering.<|endoftext|>"""

with wrapped_tokenizer as tokenizer:
        encoded_input = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)


# encoded_input={'input_ids': torch.tensor([[ 8291, 17660,   262,  1708, 13439,   422,  2679,   284,  3594,    13,
#            198, 16010,    25,  9626, 17431,    69, 14226,    89,    12,  3318,
#          17467,  2150, 36299,   567,   488,  1266,    68,  4352,   257,   385,
#            277,  9116,    77,    69,  3660,   257,   385,  3495,  1078,   316,
#            268,   371, 11033, 20080,   277, 25151,   642,   784,  1542,  7755,
#            268,    13,   198, 15823,    25,   383,  4495,  1989, 10874,   286,
#           1936,  3660,  9519, 11080,   329,   642,   532,  1542,  6506,    13,
#            775,   389,   534,  9314,  5212,   329, 19993,    11,  1641, 37134,
#             11, 11333,    11, 35401,   290, 39211,    13,   198,   198, 16010,
#             25,  5256,   374,  1155,  2853,  2574,   759,   304,  1127, 27134,
#            268,    11,   288, 44483,  7507,   271,   307,    72,   304,  7274,
#           3326,    79,    69,   385,   354,  1452, 30837,   354,   710,   312,
#           2150,   450,  1136,   918,   429,  1573,   268,  1175,    11,   288,
#            292, 14927,   308, 35410,  1976,    84,   479,   459,   380, 14226,
#           3318,   257,   794,   384,   500, 22816,   268,  1976,    84,   920,
#             69,  1142,   268,  3318,  1312, 21116,   288,  1236,   435,    82,
#            337, 11033,    67,  6607,  7128, 39683,    89, 10277,   494,   831,
#             13,   198, 15823,    25]]), 'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1]])}
with wrapped_tokenizer as tokenizer:
    new_prompt=tokenizer.batch_decode(encoded_input.input_ids)
    print("new_prompt is ",new_prompt[0])

with torch.no_grad():
    output = model(encoded_input.input_ids)
    sequences = encoded_input.input_ids
    scores = output.logits

prompt_tokens_logprobs = []

    # Append the logprob of the first token of the prompt.
prompt_tokens_logprobs.append(0.0)

# Compute logprobs of prompt tokens.
for completion_id in range(1):
    for i in range(len(sequences[completion_id]) - 1):
        logprobs = torch.nn.functional.log_softmax(scores[completion_id][i], dim=0)
        token_id=sequences[completion_id][i + 1]
        token_prob=logprobs[token_id].item()
        prompt_tokens_logprobs.append(token_prob)
        decoded_text =tokenizer.decode(token_id)
        print(f"token:  text is :{decoded_text}, prob: {token_prob}")


# with wrapped_tokenizer as tokenizer:
#     all_tokens = [[tokenizer.decode(token) for token in sequence_tokens] for sequence_tokens in sequences]
#     all_decoded_text = tokenizer.batch_decode(sequences)
# raw_completions = []
# # breakpoint()
# for decoded_text, tokens, generated_tokens_logprobs in zip(
#     all_decoded_text, all_tokens, all_generated_tokens_logprobs
# ):
    