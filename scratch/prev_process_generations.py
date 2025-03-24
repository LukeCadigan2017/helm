from src.helm.benchmark.runner import GeneratedOutputExamples
from transformers import AutoModelForCausalLM
from helm.tokenizers.huggingface_tokenizer import HuggingFaceTokenizer
from helm.common.request import (GeneratedOutput)


import json
import torch
json_file="benchmark_output/runs/run_all_evalnum_2_wmt_14_language_pair_de_en_model_distilbert_distilgpt2_follow_format_instructions_instruct_num_beams_2/wmt_14:language_pair=de-en,model=distilbert_distilgpt2/instance_generations.json"

def get_probability(model, wrapped_tokenizer, generation:GeneratedOutput):
    Translate the following sentences from German to English.
German: Der Konferenz- und Tagungsbereich besteht aus fünf modern ausgestatteten Räumen für 5 – 30 Personen.
English: The conference area consists of five modern rooms suitable for 5 - 30 persons. We are your reliable partner for conferences, family gatherings, balls, receptions and catering.

German: Er riet den Eltern eines Jungen, dessen Penis bei einer verpfuschten Beschneidung abgetrennt worden war, das Kind ganz zu kastrieren und auch seine Hoden zu entfernen und ihn dann als Mädchen großzuziehen.
English: The conference area consists of five modern rooms suitable for 5 - 30 persons. We are your reliable partner for conferences, family gatherings, balls, receptions and catering.<|endoftext|>




#     example=GeneratedOutput(**generation.examples[0])
#     print("Generation prompt: ",generation.prompt)
#     print("Generation reference: ",generation.reference)
#     print("Generation completion: ",generation.completion)
#     print("Example text         : ",example.text)
#     print()
    
#     prompt="""Translate the following sentences from German to English.
# German: Der Konferenz- und Tagungsbereich besteht aus fünf modern ausgestatteten Räumen für 5 – 30 Personen.
# English: The conference area consists of five modern rooms suitable for 5 - 30 persons. We are your reliable partner for conferences, family gatherings, balls, receptions and catering.

# German:"""+generation.prompt+"""
# English:"""

#     raw_input=prompt+example.text

#     print("raw_input is ",raw_input)
#     # print("reference is ",generation.reference)
#     with wrapped_tokenizer as tokenizer:
#         encoded_input = tokenizer(raw_input, return_tensors="pt", return_token_type_ids=False)
#         encoded_prompt = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
#     breakpoint()
#     tokens=[token["text"] for token in example.tokens]
#     print("tokens is ", tokens)
#     print("saved input is ", tokenizer.decode([tokens["text"] for token in example.tokens]))
#     print("encoded input is ",encoded_input.input_ids)

#     start=len(encoded_prompt["input_ids"][0])
    
   
#     # #So, saved token is 1 ahead
#     # for token_pos,token_id in enumerate(encoded_input.input_ids[0][start:]):
#     #     assert tokenizer(example.tokens[token_pos]['text']).input_ids[0] == token_id.item()
#     #     # print("Saved token: ",tokenizer(example.tokens[token_pos]['text']).input_ids[0], "new token: ", token_id)
#     #     # print("Saved token: ",example.tokens[token_pos]['text'], "new token: ", tokenizer.decode(token_id))

#     with torch.no_grad():
#         output = model(encoded_input["input_ids"])
#         sequences = encoded_input["input_ids"]
#         scores = output.logits

#     num_to_consider=10
#     saved_logprobs=[]
#     luke_logprobs=[]
#     new_logprobs=[]
#     for idx in range(num_to_consider):
#         saved_logprobs.append(example.tokens[idx]['logprob'])
#     print("saved_logprobs",saved_logprobs)


    # for idx in range(num_to_consider):
    #     new_token_pos=start+idx
    #     assert encoded_input.input_ids[0][new_token_pos].item() == tokenizer(example.tokens[idx]['text']).input_ids[0]
    #     token_id=encoded_input.input_ids[0][new_token_pos].item()
    #     logprobs = torch.nn.functional.log_softmax(scores[0][new_token_pos], dim=0)
    #     luke_logprobs.append(logprobs[token_id])
    # print("luke_logprobs",luke_logprobs)


    prompt_tokens_logprobs=[]

    # Compute logprobs of prompt tokens.
    completion_id=0
    for i in range(start, start+num_to_consider):
        logprobs = torch.nn.functional.log_softmax(scores[completion_id][i], dim=0)
        # breakpoint()
        prompt_tokens_logprobs.append(logprobs[sequences[completion_id][i + 1]].item())
    print("prompt_tokens_logprobs ",prompt_tokens_logprobs)




with open(json_file) as f:
    raw_generations = json.load(f)
generations= [GeneratedOutputExamples(**raw_generation) for raw_generation in raw_generations]
generation=generations[1]







#get tokens
kwargs={}
kwargs['trust_remote_code']=True
pretrained_model_name_or_path="distilbert/distilgpt2"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
wrapped_tokenizer = HuggingFaceTokenizer.create_tokenizer(pretrained_model_name_or_path)
# prompt="""Translate the following sentences from German to English.
# German: Der Konferenz- und Tagungsbereich besteht aus fünf modern ausgestatteten Räumen für 5 – 30 Personen.
# English: The conference area consists of five modern rooms suitable for 5 - 30 persons. We are your reliable partner for conferences, family gatherings, balls, receptions and catering.

# German: Er riet den Eltern eines Jungen, dessen Penis bei einer verpfuschten Beschneidung abgetrennt worden war, das Kind ganz zu kastrieren und auch seine Hoden zu entfernen und ihn dann als Mädchen großzuziehen.
# English:"""

# prompt2="""Prompt is  Translate the following sentences from German to English.
# German: Mit iPools Benutzerverwaltung können Sie all Ihre Medien- und Geschäftspartner erfassen und verwalten, die auf Ihre Musik Zugriff erhalten sollen.
# English: iPool's user administration allows you to record and administer all your media and business partners who are to have access to your music.

# German: Airbus erklärt, die konkurrierende Version des A350 befördere 350 Personen in 18 Zoll breiten Sitzen in der Touristenklasse, wobei es neun pro Reihe gibt.
# English:"""
prompt3="""Translate the following sentences from German to English.
German: Der Konferenz- und Tagungsbereich besteht aus fünf modern ausgestatteten Räumen für 5 – 30 Personen.
English: The conference area consists of five modern rooms suitable for 5 - 30 persons. We are your reliable partner for conferences, family gatherings, balls, receptions and catering.

German: Airbus erklärt, die konkurrierende Version des A350 befördere 350 Personen in 18 Zoll breiten Sitzen in der Touristenklasse, wobei es neun pro Reihe gibt.
English:"""



get_probability(model,wrapped_tokenizer, generation)


