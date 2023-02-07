##Code to use BioGPT model
from transformers import AutoTokenizer, BioGptPreTrainedModel, BioGptModel, BioGptTokenizer, BioGptForCausalLM, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large")

prompt = input("Enter a prompt: ")
input_ids = tokenizer(prompt, return_tensors="pt")  # Batch size 1
outputs = model.generate(**input_ids)
gen_txt = tokenizer.decode(outputs[0], max_new_tokens=1024,skip_special_tokens=True)

print(gen_txt)
