
from transformers import AutoTokenizer

def get_deploy_str(model_name):
  return f"""  - name: {model_name}
    model_name: {model_name}
    tokenizer_name: {model_name}
    max_sequence_length: 32768
    client_spec:
      class_name: "helm.clients.huggingface_client.HuggingFaceClient"
      args: 
        pretrained_model_name_or_path: {model_name}"""
  
def get_tokenizer(model_name):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  eos_token = tokenizer.eos_token
  bos_token = tokenizer.bos_token

  return f"""  - name: {model_name} 
    tokenizer_spec:
      class_name: "helm.tokenizers.huggingface_tokenizer.HuggingFaceTokenizer"
      args:
        pretrained_model_name_or_path: {model_name}
    end_of_text_token: "{eos_token}"
    prefix_token: "{bos_token}" """

def get_metadata(model_name,model_base, model_ext):
  return f"""  - name: {model_name}
    display_name: {model_ext}
    description: {model_ext}
    creator_organization_name: {model_base}
    access: open
    release_date: 2024-05-13
    tags: [TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, INSTRUCTION_FOLLOWING_MODEL_TAG]"""



def get_all_strings(model_base, model_ext):
  model_name=f"{model_base}/{model_ext}"
  deploy_str = get_deploy_str(model_name)
  tokenizer_str =  get_tokenizer(model_name)
  metadata_str = get_metadata(model_name, model_base, model_ext)
  return deploy_str, tokenizer_str, metadata_str


model_tuples=[]

# for model_ext in ["OLMo-2-0425-1B-Instruct","OLMo-2-0325-32B-Instruct", "OLMo-2-1124-7B-Instruct"]:
#   model_tuples.append( ( "allenai", model_ext) )


# for model_ext in ["OLMo-2-1124-7B-RM", "OLMo-2-1124-13B-RM"]:
#   model_tuples.append( ( "allenai", model_ext) )

# for model_ext in ["Meta-Llama-3-70B-Instruct"]:
#   model_tuples.append(("meta-llama", model_ext))

for model_name in ["allenai/OLMo-2-0425-1B"]:
# for model_name in "allenai/OLMoE-1B-7B-0125","allenai/OLMoE-1B-7B-0125-SFT","allenai/OLMoE-1B-7B-0125-DPO","allenai/OLMoE-1B-7B-0125-Instruct","allenai/OLMoE-1B-7B-0125-RM":

# for model_name in ["allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-13B", "allenai/OLMo-2-1124-7B-SFT ", "allenai/OLMo-2-1124-13B-SFT", "allenai/OLMo-2-1124-7B-DPO", "allenai/OLMo-2-1124-13B-DPO"]:
  split_name = model_name.strip().split("/")  
  # model_base, model_ext 
  model_tuples.append( (split_name[0], split_name[1]))
  print(f"split is ({split_name[0]},{split_name[1]})")

deploy_strs = []
token_strs=[]
metadata_strs=[]
model_names=[]
for model_tuple in model_tuples:
  model_base, model_ext = model_tuple
  deploy_str, tokenizer_str, metadata_str =  (get_all_strings(model_base, model_ext))
  deploy_strs.append(deploy_str)
  token_strs.append(tokenizer_str)
  metadata_strs.append(metadata_str)
  
  model_names.append(f"{model_base}/{model_ext}")



print("--------")
print("code ./src/helm/config/model_deployments.yaml")
print("--------")
print("\n")


print("\n\n".join(deploy_strs))

print("code ./src/helm/config/tokenizer_configs.yaml")
print("--------")
print("\n")

print("--------")

print("\n\n".join(token_strs))

print("--------")
print("code ./src/helm/config/model_metadata.yaml")
print("--------")
print("\n")
print("--------")

print("\n\n".join(metadata_strs))

print("\n".join(model_names))


# allenai/OLMo-2-0425-1B-Instruct
# allenai/OLMo-2-0325-32B-Instruct
# allenai/OLMo-2-1124-7B-Instruct
# meta-llama/Llama-3.2-1B-Instruct
# meta-llama/Llama-3.1-8B-Instruct