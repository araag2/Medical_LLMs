import os
import json
import torch
import typing
import random
import re

from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def extract_info_from_query(query : dict) -> dict:
    relevant_info = {}
    relevant_info["hypothesis"] = query["Statement"]
    relevant_info["primary_evidence"] = query["Primary_id_txt"]
    if "Secondary_id_txt" in query:
        relevant_info["secondary_evidence"] = query["Secondary_id_txt"]
    return relevant_info

def generate_query_from_prompt(text_to_replace: dict, prompt_dict: dict) -> str:
    premise = prompt_dict["primary_premise"].replace("$primary_evidence", text_to_replace["primary_evidence"])
    if "secondary_premise" in text_to_replace:
        premise += prompt_dict["secondary_premise"].replace("$secondary_evidence", text_to_replace["secondary_evidence"])
    options = prompt_dict["options"]
    
    # "$premise \n Question: Does this imply that $hypothesis? $options"
    res = prompt_dict["prompt"].replace("$premise", premise).replace("$hypothesis", text_to_replace["hypothesis"]).replace("$options", options)

    return res

ENTAILMENT_LABELS = {"Entailment", "entailment", "Yes", "yes", "y"}
CONTRADICTION_LABELS = {"Contradiction", "contradiction", "No", "no", "n"}

def textlabel_2_binarylabel(text_label: str) -> int:
    if text_label in ENTAILMENT_LABELS:
        return 1
    elif text_label in CONTRADICTION_LABELS:
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

def create_qid_prompt_label_dict(queries : dict, qrels : dict, prompts : dict) -> dict:
    queries_dict = {}
    for q_id in queries:
        queries_dict[q_id] = {}
        queries_dict[q_id]["text"] = generate_query_from_prompt(extract_info_from_query(queries[q_id]), prompts)
        queries_dict[q_id]["gold_label"] = textlabel_2_binarylabel(qrels[q_id]["Label"].strip())
    return queries_dict

def query_inference(model : object, tokenizer : object, queries : dict) -> dict:
    res_labels = {}
    with torch.inference_mode():
        for q_id in tqdm(queries):
            input_ids = tokenizer(queries[q_id]["text"][:1300] + queries[q_id]["text"][-1300:], return_tensors="pt").input_ids.to("cuda")
            #input_ids = tokenizer(queries[q_id]["text"], return_tensors="pt").input_ids.to("cuda")
            outputs = model.generate(input_ids, max_new_tokens=100)
            decoded_output = tokenizer.decode(outputs[0][input_ids[0].shape[0]:]).strip()
            decoded_output_sub = re.sub("[,!\.]+", " ", decoded_output)
            decoded_output_sub = re.sub("(\\n)+", " ", decoded_output_sub)
            decoded_output_sub = re.sub("(<\/s>)+", " ", decoded_output_sub)
            print(f'The postprocessed decoded output was {decoded_output_sub.split(" ")=}')
            res_labels[q_id] = textlabel_2_binarylabel(decoded_output_sub.split(" ")[0])
    return res_labels

def single_query_inference(model : object, tokenizer : object, prompt : str) -> str:
    with torch.inference_mode():
        print(f'{prompt=}')
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(input_ids, max_new_tokens=100)
        decoded_output = tokenizer.decode(outputs[0][input_ids[0].shape[0]:]).strip()
        decoded_output_sub = re.sub("[,!\.]+", " ", decoded_output)
        decoded_output_sub = re.sub("(\\n)+", " ", decoded_output_sub)
        decoded_output_sub = re.sub("(<\/s>)+", " ", decoded_output_sub)
        print(f'The postprocessed decoded output was {decoded_output_sub=}')
        quit()
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
        res_labels[0].append(gold_labels[q_id]["gold_label"])
        res_labels[1].append(pred_labels[q_id])

    precision = precision_score(res_labels[0], res_labels[1])
    recall = recall_score(res_labels[0], res_labels[1])
    f1 = f1_score(res_labels[0], res_labels[1])

    return {"f1" : f1, "precision" : precision, "recall" : recall}

def output_task_results(output_dir : str, model_name : str, used_set : str, results : dict):
    with safe_open_w(f'{output_dir}task_output/{model_name.split("/")[-1]}_0-shot_{used_set}-set.json') as output_file:
        output_file.write(json.dumps(results, ensure_ascii=False, indent=4))

def output_full_metrics(args : dict, full_prompt : str, used_set : str, metrics : dict):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    results = {"timestamp" : timestamp}
    for arg in vars(args):
        results[arg] = getattr(args, arg)
    results["prompt"] = full_prompt
    results["set"] = used_set
    results["metrics"] = metrics
    results["formated_metrics"] =f'| {args.model_optimize_name.split("/")[-1]}_(aplt5)_(gen)_   | {metrics["f1"]} | {metrics["precision"]} | {metrics["recall"]} | - |'

    with safe_open_w(f'{args.output_dir}ea_outputs/args_output/{timestamp}_{args.model_optimize_name.split("/")[-1]}_{used_set}-set.json') as output_file:
        output_file.write(json.dumps(results, ensure_ascii=False, indent=4))

def full_evaluate_prompt(model: object, tokenizer: object, queries: dict, qrels: dict, prompts: dict, args : object, used_set : str) -> dict:
    # Replace prompt with query info
    queries_dict = create_qid_prompt_label_dict(queries, qrels, prompts)

    # 0-shot inference from queries TODO
    pred_labels = query_inference(model, tokenizer, queries_dict)

    # Compute metrics
    metrics = calculate_metrics(pred_labels, queries_dict)
    output_full_metrics(args, prompts, used_set, metrics)

    return metrics
