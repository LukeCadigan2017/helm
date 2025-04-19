#model_metadata
model_base="prometheus-eval"
# model_ext="prometheus-7b-v2.0"
model_ext="prometheus-13b-v1.0"
model_name=f"{model_base}/{model_ext}"
eos_token="</s>"
prefix_token="<s>" 

    

print("--------")
print("code ./src/helm/config/model_deployments.yaml")
print("--------")
print("\n")
print(f"""  - name: {model_name}
    model_name: {model_name}
    tokenizer_name: {model_name}
    max_sequence_length: 32768
    client_spec:
      class_name: "helm.clients.huggingface_client.HuggingFaceClient"
      args: 
        pretrained_model_name_or_path: {model_name}""")

print("--------")
print("code ./src/helm/config/model_metadata.yaml")
print("--------")
print("\n")
print(f"""  - name: {model_name}
    display_name: {model_ext}
    description: {model_ext}
    creator_organization_name: {model_base}
    access: open
    release_date: 2024-05-13
    tags: [TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, INSTRUCTION_FOLLOWING_MODEL_TAG]""")
print("--------")


print("code ./src/helm/config/tokenizer_configs.yaml")
print("--------")
print("\n")
print(f"""  - name: 
    tokenizer_spec:
      class_name: "helm.tokenizers.huggingface_tokenizer.HuggingFaceTokenizer"
      args:
        pretrained_model_name_or_path: {model_name}
    end_of_text_token: "{eos_token}"
    prefix_token: "{prefix_token}" """)
print("--------")
