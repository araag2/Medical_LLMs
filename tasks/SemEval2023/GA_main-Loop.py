import os
import argparse
import json
import torch
import typing
import random
import itertools 
import re
import GA_evaluation

from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import GPTQConfig, LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM

#if "SLURM_JOB_ID" not in os.environ:
#    device = "CPU"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def ea_generate_prompts_OAI(model_name: str, ea_prompt: str, prompt_1: str, prompt_2: str) -> str:
    new_prompt = ea_prompt.replace("$prompt_1", prompt_1).replace("$prompt_2", prompt_2)
    client = OpenAI()

    response = client.chat.completions.create(
               model=model_name,
               messages=[
               {"role": "system", "content": "You are an intelligent model that serves to combine and create better prompts from existing prompt. Follow the instructions carrefully."},
               {"role": "user", "content": new_prompt}])
    print(response)
    return response.choices.message.content

def ea_generate_prompts_qCammel(model : object, tokenizer : object, ea_prompt : str, prompt_1 : str, prompt_2 : str) -> str:
    new_prompt = ea_prompt.replace("<prompt_1>", prompt_1).replace("<prompt_2>", prompt_2)
    return GA_evaluation.single_query_inference(model, tokenizer, new_prompt)

def ea_generate_pairs(curr_parent_prompts: dict, base_prompt: dict) -> dict:
    pairs = {}
    for id_1 in curr_parent_prompts:
        content_1 = curr_parent_prompts[id_1]
        for id_2 in curr_parent_prompts:
            content_2 = curr_parent_prompts[id_2]
            if id_1 != id_2:
                pairs[id_1+"_"+id_2] = {"prompt_1" : content_1, "prompt_2" : content_2, "base_prompt" : base_prompt["base_prompt"], "primary_premise" : base_prompt["primary_premise"], "secondary_premise" : base_prompt["secondary_premise"], "options" : base_prompt["options"]}
    return pairs

def sort_prompts_by_score(prompt_metrics_1: dict, prompt_metrics_2: dict, min_precison : float, max_recall : float) -> dict:
    eligible_1 = prompt_metrics_1["precision"] >= min_precison and prompt_metrics_1["recall"] <= max_recall
    eligible_2 = prompt_metrics_2["precision"] >= min_precison and prompt_metrics_2["recall"] <= max_recall

    if eligible_1 and eligible_2:
        return eligible_1["f1"] - eligible_2["f1"]
    elif eligible_1:
        return 1
    elif eligible_2: 
        return -1
    return eligible_1["f1"] - eligible_2["f1"]

def tokenize_and_gen(model : object, tokenizer : object, prompt : str, max_len : int) -> str:
    tokenized = tokenizer(prompt, return_tensors="pt")
    tokenized["input_ids"] = tokenized.input_ids.to(device="cuda")
    tokenized["attention_mask"] = tokenized.attention_mask.to(device="cuda")
    outputs =  model.generate(**tokenized, max_new_tokens=max_len, top_k = 5, do_sample=True)
    return tokenizer.decode(outputs[0][tokenized["input_ids"].shape[1]:]).strip()

def mutate_parent_prompts(model : object, tokenizer : object, mutate_prompt : str, relevant_segments : str, prompts_to_mutate : list[dict]) -> list[dict]:
    mutated_prompts = []
    with torch.inference_mode():
        for prompt in prompts_to_mutate:
            prompt_dict = {"prompt" : "", "prompt_partions" : [part for part in prompt["prompt_partions"]], "id" : f'mutated_{prompt["id"]}'}

            for segment in relevant_segments:
                mutate_str = mutate_prompt.replace("$prompt", prompt_dict["prompt_partions"][segment])

                decoded_output = tokenize_and_gen(model, tokenizer, mutate_str, len(prompt_dict["prompt_partions"][segment])+ 10)
                decoded_output = decoded_output if ":" not in decoded_output else decoded_output[decoded_output.index(":")+2:]
                decoded_output_sub = re.sub("(<\/s>)+", " ", decoded_output)
                decoded_output_sub = re.sub("\"", "", decoded_output_sub)
                print(f'Mutation output was {decoded_output_sub=}')
                prompt_dict["prompt_partions"][segment] = decoded_output_sub

            prompt_dict["prompt"] = "\n\n".join(prompt_dict["prompt_partions"]) 
            mutated_prompts.append(prompt_dict)
    return mutated_prompts

def combine_curr_prompts(model : object, tokenizer : object, combine_prompt : str, relevant_segments : str, prompts_to_combine : list[dict], n_combinations : int) -> list[dict]:
    # Random pairs of prompts
    random_pairs = list(itertools.combinations(prompts_to_combine, 2))
    random.shuffle(random_pairs)
    random_pairs = random_pairs[:n_combinations]

    combined_prompts = []
    with torch.inference_mode():
        for prompt_1, prompt_2 in random_pairs:
            prompt_dict = {"prompt" : "", "prompt_partions" : [part for part in prompt_1["prompt_partions"]], "id" : f'mutated_{prompt_2["id"]}'}

            for segment in relevant_segments:
                combine_str = combine_prompt.replace("$prompt_1", prompt_1["prompt_partions"][segment]).replace("$prompt_2", prompt_2["prompt_partions"][segment])

                max_len = max(len(prompt_1["prompt_partions"][segment]), len(prompt_2["prompt_partions"][segment])) + 10

                decoded_output = tokenize_and_gen(model, tokenizer, combine_str, max_len)
                decoded_output = decoded_output if ":" not in decoded_output else decoded_output[decoded_output.index(":")+2:]
                decoded_output_sub = re.sub("(<\/s>)+", " ", decoded_output)
                decoded_output_sub = re.sub("\"", "", decoded_output_sub)
                print(f'Combination output was {decoded_output_sub=}')

                prompt_dict["prompt_partions"][segment] = decoded_output_sub

            prompt_dict["prompt"] = "\n\n".join(prompt_dict["prompt_partions"]) 
            combined_prompts.append(prompt_dict)
    return combined_prompts


def main():
    parser = argparse.ArgumentParser()

    #TheBloke/Llama-2-70B-Chat-GPTQ
    #parser.add_argument('--model_optimize_name', type=str, help='name of the model used to fine-tune prompts for', default='TheBloke/qCammel-70-x-GPTQ')

    parser.add_argument('--model_name', type=str, help='name of the model used to generate and combine prompts', default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--model', type=str, help='name of the model used to generate and combine prompts', default='mistralai/Mistral-7B-Instruct-v0.2')

    used_set = "train" # train | dev | test

    # Path to corpus file
    parser.add_argument('--dataset_path', type=str, help='path to corpus file', default="../../datasets/SemEval2024/CT_corpus.json")

    # Path to queries, qrels and prompt files
    parser.add_argument('--queries', type=str, help='path to queries file', default=f'queries/queries2024_{used_set}.json')
    parser.add_argument('--qrels', type=str, help='path to qrels file', default=f'qrels/qrels2024_{used_set}.json')
    # "prompts/T5prompts.json"
    parser.add_argument('--prompts', type=str, help='path to prompts file', default="prompts/EA_Mistral_Prompts.json")

    # Evaluation metrics to use 
    #
    # Model parameters
    #
    # LLM generation parameters
    #

    # Output directory
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="outputs/ea_output/")

    # GA parameters
    parser.add_argument('--n_iterations', type=int, help='number of iterations to run GA on', default=3)
    parser.add_argument('--n_prompts', type=int, help='number of prompts to generate per iteration', default=15)
    parser.add_argument('--top_k', type=int, help='number of prompts keep for future generations', default=5)
    parser.add_argument('--combinations', type=int, help='number of combinations to generate', default=15)
    parser.add_argument('--metric', type=str, help='metric to keep top_k prompts of previous iteration', default="f1_macro")
    parser.add_argument('--min_precision', type=float, help='minimum precision for a prompt to be considered', default=0.50)
    parser.add_argument('--max_recall', type=float, help='maximum recall for a prompt to be considered', default=0.92)

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    #model.quantize_config = GPTQConfig(bits=4, exllama_config={"version":2}, desc_act=True)
    #model = exllama_set_max_input_length(model, 4096)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset, queries, qrels and prompts
    queries = json.load(open(args.queries))
    qrels = json.load(open(args.qrels))
    prompts = json.load(open(args.prompts))

    relevant_prompt_segments = [0, 1, 4, 6]

    curr_parent_prompts = [ {key : prompt[key] for key in prompt if key in ["prompt", "metrics", "prompt_partions", "id"]} for prompt in prompts["scores_precision"][:args.top_k]]

    for i in tqdm(range(1, int(args.n_iterations)+1)):
        # Mutate current prompts, generating top_k new prompts
        curr_prompts = mutate_parent_prompts(model, tokenizer, prompts["mutate_prompt"], relevant_prompt_segments, curr_parent_prompts)

        # Combine current prompts, generating new prompts
        curr_prompts = combine_curr_prompts(model, tokenizer, prompts["combine_prompt"], relevant_prompt_segments, curr_prompts + curr_parent_prompts, args.combinations)

        # Evaluate new prompts
        #print("\n\n\n\nEND PROMPTS OF ITER " + str(i))
        #with safe_open_w(f'{args.output_dir}prompt-comb_iter-{i}.json') as f:
        #    json.dump(curr_prompts, f, indent=4)

        for prompt in tqdm(curr_prompts):
            prompt["metrics"] = GA_evaluation.full_evaluate_prompt(model, tokenizer, queries, qrels, prompt["id"], prompt["prompt"], args, used_set)["metrics"]
        
        curr_prompts = curr_prompts + curr_parent_prompts

        # Sort curr_prompts by score
        curr_prompts.sort(key=lambda x: x["metrics"][args.metric], reverse=True)
        # Output iter res to file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        with safe_open_w(f'{args.output_dir}{timestamp}_EA-Mistral_iter-{i}.json') as f:
            json.dump(curr_prompts, f, indent=4)
        curr_parent_prompts = curr_prompts[:args.top_k]

if __name__ == '__main__':
    main()