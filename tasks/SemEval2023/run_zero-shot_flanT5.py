import os
import argparse
import json
import torch
import typing
import random

from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import T5Tokenizer, T5ForConditionalGeneration

#if "SLURM_JOB_ID" not in os.environ:
#    device = "CPU"

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
    res = prompt_dict["baseline_prompt"].replace("$premise", premise).replace("$hypothesis", text_to_replace["hypothesis"]).replace("$options", options)

    return res

ENTAILMENT_LABELS = {"Entailment", "entailment", "Yes", "yes", "y"}
CONTRADICTION_LABELS = {"Contradiction", "contradiction", "No", "no", "n"}

def textlabel_2_binarylabel(text_label: str) -> int:
    if text_label in ENTAILMENT_LABELS:
        return 1
    elif text_label in CONTRADICTION_LABELS:
        return 0
    print("Executed a Random choice because the label was not found.")
    return random.randint(0,1)

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

def query_inference(model : object, tokenizer : object, queries : dict) -> dict:
    res_labels = {}
    with torch.inference_mode():
        for q_id in tqdm(queries):
            input_ids = tokenizer(queries[q_id]["text"], return_tensors="pt").input_ids.to("cuda")
            outputs = model.generate(input_ids)
            res_labels[q_id] = textlabel_2_binarylabel(tokenizer.decode(outputs[0])[5:][:-4].strip())
    return res_labels

def calculate_metrics(pred_labels : dict, gold_labels : dict) -> dict:
    res_labels = [[],[]]
    for q_id in pred_labels:
        res_labels[0].append(gold_labels[q_id]["gold_label"])
        res_labels[1].append(pred_labels[q_id])

    accuracy = accuracy_score(res_labels[0], res_labels[1])
    precision = precision_score(res_labels[0], res_labels[1])
    recall = recall_score(res_labels[0], res_labels[1])
    f1 = f1_score(res_labels[0], res_labels[1])

    return {"accuracy" : accuracy, "precision" : precision, "recall" : recall, "f1" : f1}

def main():
    parser = argparse.ArgumentParser()
    # Model name to use (downloaded from huggingface)
    parser.add_argument('--model_name', type=str, help='name of the T5 model used', default='google/flan-t5-base')
    
    # Path to corpus file
    parser.add_argument('--dataset_path', type=str, help='path to corpus file', default="../../datasets/SemEval2023/CT_corpus.json")
    # Path to queries, qrels and prompt files
    parser.add_argument('--queries', type=str, help='path to queries file', default="queries/queries2023_train.json")
    parser.add_argument('--qrels', type=str, help='path to qrels file', default="qrels/qrels2023_train.json")
    parser.add_argument('--prompts', type=str, help='path to prompts file', default="prompts/T5prompts.json")

    # Evaluation metrics to use 
    #
    # Model parameters
    #
    # LLM generation parameters
    #

    # Output directory
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="../outputs/")
    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name, device_map="auto")

    # Load dataste, queries, qrels and prompts
    #dataset = json.load(open(args.dataset_path))
    queries = json.load(open(args.queries))
    qrels = json.load(open(args.qrels))
    prompts = json.load(open(args.prompts))

    # Replace prompt with query info
    queries_dict = {}
    for q_id in queries:
        queries_dict[q_id] = {}
        queries_dict[q_id]["text"] = generate_query_from_prompt(extract_info_from_query(queries[q_id]), prompts)
        queries_dict[q_id]["gold_label"] = textlabel_2_binarylabel(qrels[q_id]["Label"].strip())

    # 0-shot inference from queries
    pred_labels = query_inference(model, tokenizer, queries_dict)

    # Compute metrics
    metrics = calculate_metrics(pred_labels, queries_dict)

    # Format to SemEval2023 format
    formated_results = label_2_SemEval2023(pred_labels)

    print(metrics)

    # Output Res
    with safe_open_w((f'{args.output_dir}{args.model_name}_0-shot_train-set.json')) as output_file:
        output_file.write(json.dumps(formated_results, ensure_ascii=False, indent=4))

if __name__ == '__main__':
    main()