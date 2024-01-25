import os
import json
import argparse
import torch
import re

from negate import Negator
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM

def safe_open_w(path: str):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='name of the model used to fine-tune prompts for', default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--input', type=str, help='path to input dir/file', default='qrels/qrels2024_train-dev_manual-Expand.json')
    parser.add_argument('--output', type=str, help='path to output dir/file', default='qrels/qrels2024_train-dev_manual-Expand-2.json')
    #parser.add_argument('--corpus', type=str, help='path to CT Corpus', default='CT_json/SemEval_CT-corpus.json')
    args = parser.parse_args() 

    qrels = json.load(open(args.input))

    output_dict = {}

    prompt = '<s>[INST]Re-write the following statement without altering its semantic meaning, present in between quotes. \n\n"$statement"\n\nOnly answer with the re-write, without providing any additional explanation.[/INST]'

    Entailment = 0
    Contradiction = 0

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    for qrel in tqdm(qrels):
        if "TO DO" not in qrels[qrel]["Statement"]:
            output_dict[qrel] = {}
            for field in qrels[qrel]:
                output_dict[qrel][field] = qrels[qrel][field]
            if qrels[qrel]["Label"] == "Entailment":
                Entailment += 1
            elif qrels[qrel]["Label"] == "Contradiction":
                Contradiction += 1

    needed_entailment = Contradiction - Entailment #891
    generated = 0

    for qrel in tqdm(list(output_dict.keys())):
       if output_dict[qrel]["Label"] == "Entailment":
            with torch.inference_mode():
                sent_prompt = prompt.replace("$statement", output_dict[qrel]["Statement"])                
                for i in range(5):                       
                    tokenized = tokenizer(sent_prompt, return_tensors="pt")
                    tokenized["input_ids"] = tokenized.input_ids.to(device="cuda")
                    tokenized["attention_mask"] = tokenized.attention_mask.to(device="cuda")
                    outputs =  model.generate(**tokenized, max_new_tokens=len(output_dict[qrel]["Statement"]) // 4 + 20, top_k = 5 + i*10, do_sample=True, temperature = 1.0 + i*0.4, pad_token_id=tokenizer.eos_token_id)
            
                    decoded_output = tokenizer.decode(outputs[0][tokenized["input_ids"].shape[1]:]).strip()
                    decoded_output_sub = re.sub("(\\n)+", " ", decoded_output)
                    decoded_output_sub = re.sub("(<\/s>)+", " ", decoded_output_sub)
                    decoded_output_sub = re.sub('\"', "", decoded_output_sub)
            
                    #print(f'{qrel}_{i}: {decoded_output_sub=}')
                    output_dict[f'{qrel}_{i}'] = {}
                    for field in qrels[qrel]:
                        output_dict[f'{qrel}_{i}'][field] = qrels[qrel][field]
                    
                    output_dict[f'{qrel}_{i}']["Statement"] = decoded_output_sub  
                    generated += 1
            if generated >= needed_entailment:
                break   
        
    with safe_open_w(args.output) as f:
        json.dump(output_dict, f, indent=4)