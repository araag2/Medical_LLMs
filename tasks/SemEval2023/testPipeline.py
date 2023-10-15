from transformers import AutoTokenizer
import transformers
import torch

model = "TheBloke/qCammel-13-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
pipeline = transformers.pipeline("text-generation", model=model, torch_dtype=torch.float16, device_map="auto", revision="main")

prompt = """
Consider evidence from two different clinical trials.

The primary clinical trial evidence lists the following:

INTERVENTION 1: 
* Letrozole, Breast Enhancement, Safety.
* Single arm of healthy postmenopausal women to have two breast MRI (baseline and post-treatment). Letrozole of 12.5 mg/day is given for three successive days just prior to the second MRI.

The secondary clinical trial evidence lists the following:

INTERVENTION 1: 
* Healthy Volunteers.
* Healthy women will be screened for Magnetic Resonance Imaging (MRI) contraindications, and then undergo contrast injection, and SWIFT acquisition.
* Magnetic resonance imaging: Patients and healthy volunteers will be first screened for MRI contraindications. The SWIFT MRI workflow will be performed.

What is the likelihood of the evidence implying that the primary trial and the secondary trial both used MRI for their interventions?
Provide an answer in the range between one and five.

"""
sequences = pipeline(prompt, top_k=5, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_new_tokens=4096, do_sample=True)

for seq in sequences: print(f"Result: {seq['generated_text']}")