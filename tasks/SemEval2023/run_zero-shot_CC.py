import os
import argparse
import json
import torch
import typing
import random
import re

from datetime import datetime
from auto_gptq import exllama_set_max_input_length
from tqdm import tqdm
from huggingface_hub import login
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import GPTQConfig, LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM

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

def generate_query_from_prompt(text_to_replace: dict, prompt_dict: dict, prompt_id : str = None) -> str:
    premise = prompt_dict["primary_premise"].replace("$primary_evidence", text_to_replace["primary_evidence"])
    if "secondary_premise" in text_to_replace:
        premise += prompt_dict["secondary_premise"].replace("$secondary_evidence", text_to_replace["secondary_evidence"])
    options = prompt_dict["options"]
    
    base_prompt = None
    if prompt_id != None:
        base_prompt = prompt_dict["base_prompt"].replace("$new_prompt", prompt_dict[prompt_id])
    else:
        base_prompt = prompt_dict["base_prompt"].replace("$new_prompt", prompt_dict["10"])
    res = base_prompt.replace("$premise", premise).replace("$hypothesis", text_to_replace["hypothesis"]).replace("$options", options)

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

def create_qid_prompt_label_dict(queries : dict, qrels : dict, prompts : dict, prompt_id : str) -> dict:
    queries_dict = {}
    for q_id in queries:
        queries_dict[q_id] = {}
        queries_dict[q_id]["text"] = generate_query_from_prompt(extract_info_from_query(queries[q_id]), prompts, prompt_id)
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

    results["formated_metrics"] =f'| {args.model_name.split("/")[-1]}_(aplt5)_(gen)_{args.prompt_id}   | {metrics["f1"]:.2f} | {metrics["precision"]:.2f} | {metrics["recall"]:.2f} | - |'

    with safe_open_w(f'{args.output_dir}args_output/{timestamp}_{args.model_name.split("/")[-1]}_{used_set}-set.json') as output_file:
        output_file.write(json.dumps(results, ensure_ascii=False, indent=4))

def main():
    parser = argparse.ArgumentParser()
    # Model name to use (downloaded from huggingface)
    #[Asclepius-Llama2-13B](https://huggingface.co/starmpcc/Asclepius-Llama2-13B)
    #[Asclepius-13B-GPTQ](https://huggingface.co/TheBloke/Asclepius-13B-GPTQ)
    #[qCammel-13-GPTQ](https://huggingface.co/TheBloke/qCammel-13-GPTQ)
    #[qCammel-13B-Combined-Data-GPTQ](https://huggingface.co/TheBloke/CAMEL-13B-Combined-Data-GPTQ)
    #[qCammel-13B-Role-Playing-GPTQ](https://huggingface.co/TheBloke/CAMEL-13B-Role-Playing-Data-GPTQ)
    #[qCammel-70-x-GPTQ](https://huggingface.co/TheBloke/qCammel-70-x-GPTQ)
    #[qCammel-70-x-GPTQ-gptq-3bit-128g](https://huggingface.co/TheBloke/qCammel-70-x-GPTQ/tree/gptq-3bit-128g-actorder_True)

    # '/user/home/aguimas/data/PhD/models/TheBloke-qCammel-70-x-GPTQ-gptq-3bit-128g/'
    # 'TheBloke/CAMEL-13B-Combined-Data-GPTQ'
    parser.add_argument('--model_name', type=str, help='name of the T5 model used', default='TheBloke/qCammel-70-x-GPTQ')

    used_set = "dev" # train | dev | test

    # Path to corpus file
    parser.add_argument('--dataset_path', type=str, help='path to corpus file', default="../../datasets/SemEval2023/CT_corpus.json")
    # Path to queries, qrels and prompt files
    parser.add_argument('--queries', type=str, help='path to queries file', default=f'queries/queries2023_{used_set}.json')
    parser.add_argument('--qrels', type=str, help='path to qrels file', default=f'qrels/qrels2023_{used_set}.json')
    # "prompts/T5prompts.json"
    parser.add_argument('--prompts', type=str, help='path to prompts file', default="prompts/GA_generated-prompts.json")
    parser.add_argument('--prompt_id', type=str, help='id of the prompt to use', default='8_10')

    # Evaluation metrics to use 
    #
    # Model parameters
    #
    # LLM generation parameters
    #

    # Output directory
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="outputs/")
    args = parser.parse_args()

    # Login to huggingface
    #login(token=os.environ["HUGGINGFACE_TOKEN"])

    # TODO: Check LlamaModel here too
    #model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", quantization_config = GPTQConfig(bits=4, exllama_config={"version":2}, desc_act=True))
    model = LlamaForCausalLM.from_pretrained(args.model_name, device_map="auto")
    model.quantize_config = GPTQConfig(bits=4, exllama_config={"version":2}, desc_act=True)
    model = exllama_set_max_input_length(model, 4096)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    #model = AutoModelForCausalLM.from_pretrained(args.model_name,
    #                                            torch_dtype=torch.float16,
    #                                            device_map="auto",
    #                                            revision="gptq-3bit--1g-actorder_True")
    #tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    #model = exllama_set_max_input_length(model, 4096)

    # Load dataset, queries, qrels and prompts
    #dataset = json.load(open(args.dataset_path))
    queries = json.load(open(args.queries))
    qrels = json.load(open(args.qrels))
    prompts = json.load(open(args.prompts))

    # Replace prompt with query info
    queries_dict = create_qid_prompt_label_dict(queries, qrels, prompts, args.prompt_id)

    # 0-shot inference from queries TODO
    pred_labels = query_inference(model, tokenizer, queries_dict)

    # Compute metrics
    metrics = calculate_metrics(pred_labels, queries_dict)
    output_full_metrics(args, prompts, used_set, metrics)

    # Format to SemEval2023 format
    formated_results = label_2_SemEval2023(pred_labels)
    #output_task_results(args.output_dir, args.model_name, used_set, formated_results)

if __name__ == '__main__':
    main()