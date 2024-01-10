import os
import json
import torch
import typing
import random
import re
import argparse

from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForCausalLM


def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def extract_info_from_query(query : dict) -> dict:
    relevant_info = {}
    relevant_info["hypothesis"] = query["Statement"]
    relevant_info["primary_evidence"] = query["Primary_id_txt"]
    relevant_info["secondary_evidence"] = query["Secondary_id_txt"] if "Secondary_id_txt" in query else ""
    return relevant_info

def generate_query_from_prompt(text_to_replace: dict, prompt: str) -> str:
    prompt = prompt.replace("$primary_evidence", text_to_replace["primary_evidence"])
    prompt = prompt.replace("$secondary_evidence", text_to_replace["secondary_evidence"])
    prompt = prompt.replace("$hypothesis", text_to_replace["hypothesis"])
    return prompt

ENTAILMENT_LABELS = {"entailment", "yes", "y"}
CONTRADICTION_LABELS = {"contradiction", "no", "not", "n"}

def textlabel_2_binarylabel(text_label: list[str]) -> int:
    for label in text_label:
        if label.lower() in ENTAILMENT_LABELS:
            return 1
        elif label.lower() in CONTRADICTION_LABELS:
            return 0
    print(f'Text label: [{text_label=}.] This label executed a Random choice because the label was not found.')
    #return random.randint(0,1)
    return 1

def label_2_SemEval2023(labels : dict) -> dict:
    res = {}
    for q_id in labels:
        pred = "None" # random.choice(["Entailment", "Contradiction"])
        if labels[q_id] == 1:
            pred = "Entailment"
        elif labels[q_id] == 0:
            pred = "Contradiction"
        res[q_id] = {"Prediction" : pred}
    return res

def create_qid_prompt_label_dict(queries : dict, qrels : dict, prompt : str) -> dict:
    queries_dict = {}
    for q_id in queries:
        queries_dict[q_id] = { 
            "text" : generate_query_from_prompt(extract_info_from_query(queries[q_id]), prompt), 
            "gold_label" : textlabel_2_binarylabel([qrels[q_id]["Label"].strip()])
        }
    return queries_dict

def query_inference(model : object, tokenizer : object, queries : dict) -> dict:
    res_labels = {}
    with torch.inference_mode():
        for q_id in tqdm(queries):
            input_ids = tokenizer(queries[q_id]["text"], return_tensors="pt").input_ids.to("cuda")
            #input_ids = tokenizer(queries[q_id]["text"], return_tensors="pt").input_ids.to("cuda")
            outputs = model.generate(input_ids, max_new_tokens=30, top_k = 5, do_sample=True)
            decoded_output = tokenizer.decode(outputs[0][input_ids[0].shape[0]:]).strip()
            decoded_output_sub = re.sub("[,!\.]+", " ", decoded_output)
            decoded_output_sub = re.sub("(\\n)+", " ", decoded_output_sub)
            decoded_output_sub = re.sub("(<\/s>)+", " ", decoded_output_sub)
            print(f'The postprocessed decoded output was {decoded_output_sub.split(" ")[:10]=}')
            res_labels[q_id] = textlabel_2_binarylabel(decoded_output_sub.split(" ")[:10])
    return res_labels

def single_query_inference(model : object, tokenizer : object, prompt : str) -> str:
    with torch.inference_mode():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(input_ids, do_sample=True, top_k = 10, max_new_tokens=50)
        decoded_output = tokenizer.decode(outputs[0][input_ids[0].shape[0]:]).strip()
        decoded_output_sub = re.sub("[,!\.]+", " ", decoded_output)
        decoded_output_sub = re.sub("(\\n)+", " ", decoded_output_sub)
        decoded_output_sub = re.sub("(<\/s>)+", " ", decoded_output_sub)
        print(f'The postprocessed decoded output was {decoded_output_sub=}')
    return decoded_output_sub

def debug_query_inference(model : object, tokenizer : object, queries : dict) -> dict:
    res_labels = {}
    with torch.inference_mode():
        print(f'Entered query_inference')
        for q_id in tqdm(list(queries.keys())[:10]):
            #print(f'length of query is {len(queries[q_id]["text"])}')
            #input_ids = tokenizer(queries[q_id]["text"][:2000], return_tensors="pt").input_ids.to("cuda")
            input_ids = tokenizer(queries[q_id]["text"], return_tensors="pt").input_ids.to("cuda")
            outputs = model.generate(input_ids, max_new_tokens=100)
            decoded_output = tokenizer.decode(outputs[0][input_ids[0].shape[0]:]).strip()
            print(f'The decoded output of {q_id=} was {decoded_output=}')
            #decoded_output_sub = re.sub("[,!\.]*(\\n)*(<\/s>)*", " ", decoded_output).split(" ")[0] #TODO: This is removing the s characters from the output
            decoded_output = re.sub("[,!\.]*(\\n)*(<\/s>)*", " ", decoded_output)
            print(f'The postprocessed decoded output of {q_id=} was {decoded_output=}')
            res_labels[q_id] = textlabel_2_binarylabel(decoded_output.split(" ")[0])
    return res_labels 

def calculate_metrics(pred_labels : dict, gold_labels : dict) -> dict:
    res_labels = [[],[]]
    for q_id in pred_labels:
        print(f'{gold_labels[q_id]["gold_label"]=} {pred_labels[q_id]=}')
        res_labels[0].append(gold_labels[q_id]["gold_label"])
        res_labels[1].append(pred_labels[q_id])

    precision = precision_score(res_labels[0], res_labels[1])
    recall = recall_score(res_labels[0], res_labels[1])
    f1 = f1_score(res_labels[0], res_labels[1])

    return {"f1" : f1, "precision" : precision, "recall" : recall}

def output_task_results(output_dir : str, model_name : str, used_set : str, results : dict):
    with safe_open_w(f'{output_dir}task_output/{model_name.split("/")[-1]}_0-shot_{used_set}-set.json') as output_file:
        output_file.write(json.dumps(results, ensure_ascii=False, indent=4))

def output_full_metrics(args : dict, prompt_id : str, full_prompt : str, used_set : str, metrics : dict):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    results = {"timestamp" : timestamp}
    for arg in vars(args):
        results[arg] = getattr(args, arg)
    results["prompt"] = full_prompt
    results["set"] = used_set
    results["metrics"] = metrics
    results["formated_metrics"] =f'| {args.model.split("/")[-1]}-{prompt_id}   | {metrics["f1"]} | {metrics["precision"]} | {metrics["recall"]} | - |'

    with safe_open_w(f'{args.output_dir}combination_output/{timestamp}_{args.model.split("/")[-1]}_{used_set}-set.json') as output_file:
        output_file.write(json.dumps(results, ensure_ascii=False, indent=4))

def full_evaluate_prompt(model: object, tokenizer: object, queries: dict, qrels: dict, prompt_id : str, prompt: str, args : object, used_set : str) -> dict:
    # Replace prompt with query info
    queries_dict = create_qid_prompt_label_dict(queries, qrels, prompt)

    # 0-shot inference from queries TODO
    pred_labels = query_inference(model, tokenizer, queries_dict)

    # Compute metrics
    metrics = calculate_metrics(pred_labels, queries_dict)
    output_full_metrics(args, prompt_id, prompt, used_set, metrics)

    return metrics

def generate_pos_prompts(mistral_prompts : dict):
    prompt_combinations = { "base_mistral_prompts" : {field : mistral_prompts[field] for field in mistral_prompts}, "combination_prompts" : {}}

    for task_id, task in mistral_prompts["task_descriptions"].items():
        for ctr_id, ctr in mistral_prompts["ctr_descriptions"].items():
            for statement_id, statement in mistral_prompts["statement_descriptions"].items():
                for option_id, option in mistral_prompts["option_descriptions"].items():
                    combination = mistral_prompts["task_template_prompt_comparison"].replace("$task_description", task).replace("$ctr_description", ctr).replace("$statement_description", statement).replace("$option_description", option)

                    prompt_combinations["combination_prompts"][f'{task_id}_{ctr_id}_{statement_id}_{option_id}'] = combination

    with safe_open_w(f'prompts/MistralPromptsCombination.json') as output_file:
        output_file.write(json.dumps(prompt_combinations, ensure_ascii=False, indent=4))

    return prompt_combinations


def main():
    parser = argparse.ArgumentParser()

    #TheBloke/Llama-2-70B-Chat-GPTQ
    parser.add_argument('--model', type=str, help='name of the model used to fine-tune prompts for', default='mistralai/Mistral-7B-Instruct-v0.2')

    used_set = "dev" # train | dev | test

    # Path to corpus file
    parser.add_argument('--dataset_path', type=str, help='path to corpus file', default="../../datasets/SemEval2023/CT_corpus.json")

    # Path to queries, qrels and prompt files
    parser.add_argument('--queries', type=str, help='path to queries file', default=f'queries/queries2023_{used_set}.json')
    parser.add_argument('--qrels', type=str, help='path to qrels file', default=f'qrels/qrels2023_{used_set}.json')
    # "prompts/T5prompts.json"
    parser.add_argument('--prompts', type=str, help='path to prompts file', default="prompts/MistralPrompts.json")

    # Output directory
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="outputs/")

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    #model.quantize_config = GPTQConfig(bits=4, exllama_config={"version":2}, desc_act=True)
    #model = exllama_set_max_input_length(model, 4096)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load dataset, queries, qrels and prompts
    queries = json.load(open(args.queries))
    qrels = json.load(open(args.qrels))
    prompts = json.load(open(args.prompts))

    combination_prompts = generate_pos_prompts(prompts)

    for prompt_id, prompt in tqdm(combination_prompts["combination_prompts"].items()):
        full_evaluate_prompt(model, tokenizer, queries, qrels, prompt_id, prompt, args, used_set)
    

if __name__ == '__main__':
    main()