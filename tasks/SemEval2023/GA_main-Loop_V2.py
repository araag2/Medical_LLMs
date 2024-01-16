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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def process_decoded_output(decoded_output : str) -> str:
    decoded_output = decoded_output if ":" not in decoded_output else decoded_output[decoded_output.index(":")+2:]
    decoded_output_sub = re.sub("(<\/s>)+", " ", decoded_output)
    return re.sub("\"", "", decoded_output_sub)

def tokenize_and_gen(model : object, tokenizer : object, prompt : str, max_len : int) -> str:
    with torch.inference_mode():
        tokenized = tokenizer(prompt, return_tensors="pt")
        tokenized["input_ids"] = tokenized.input_ids.to(device="cuda")
        tokenized["attention_mask"] = tokenized.attention_mask.to(device="cuda")
        outputs =  model.generate(**tokenized, max_new_tokens=max_len, top_k = 5, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return process_decoded_output(tokenizer.decode(outputs[0][tokenized["input_ids"].shape[1]:]).strip())

def combine_curr_prompts(model : object, tokenizer : object, combine_prompt : str, relevant_segments : str, prompts_to_combine : list[dict]) -> list[tuple]:
    res = [[] for _ in range(len(relevant_segments))]

    for i, segment in enumerate(relevant_segments):
        seg_1 =  prompts_to_combine[0]["prompt_partions"][segment]
        seg_2 =  prompts_to_combine[1]["prompt_partions"][segment]

        if seg_1 == seg_2:
            # The bool is used to indicate if the segment is a combination of two prompts or not
            res[i] = [(seg_1, False)]

        else:
            combine_str = combine_prompt.replace("$prompt_1", seg_1).replace("$prompt_2", seg_2)
            max_len = max(len(seg_1), len(seg_2)) + 50

            res[i] = [(seg_1, False), (seg_2, False), (tokenize_and_gen(model, tokenizer, combine_str, max_len), True)]

    return res

def mutate_parent_prompts(model : object, tokenizer : object, mutate_prompt : str, relevant_segments : str, prompts_to_mutate : list[dict]) -> list[dict]:
    res = [[] for _ in range(len(relevant_segments))]

    for i, segment in enumerate(relevant_segments):
        seg_1 =  prompts_to_mutate[0]["prompt_partions"][segment]
        seg_2 =  prompts_to_mutate[1]["prompt_partions"][segment]

        if seg_1 == seg_2:
            mutate_str = mutate_prompt.replace("$prompt", seg_1)
            max_len = len(seg_1) + 50
            for _ in range(4):
                # The bool is used to indicate if the segment is a mutation of two prompts or not
                res[i].append((tokenize_and_gen(model, tokenizer, mutate_str, max_len), True))

        else:
            res[i].append((tokenize_and_gen(model, tokenizer, mutate_prompt.replace("$prompt", seg_1), len(seg_1) + 50), True))
            res[i].append((tokenize_and_gen(model, tokenizer, mutate_prompt.replace("$prompt", seg_2), len(seg_2) + 50), True))

    return res

def sample_segments(curr_iter_segments : list[list[str]], relevant_segments : list[int], N : int) -> list[list[str]]:
    list_segments = [iter_segments for iter_segments in curr_iter_segments]
    sample_probabilities = []

    for i,iter_segments in enumerate(list_segments):
        iter_probabilities = ([], [])
        for j, segment in enumerate(iter_segments):
            iter_probabilities[0].append((i,j))
            iter_probabilities[1].append(3 if segment[1] else 1)
        sample_probabilities.append(iter_probabilities)

    sampled_segments_set = set()

    while len(sampled_segments_set) < N:
        sample = [] 
        for i in range(len(relevant_segments)):
            sample.append(random.choices(sample_probabilities[i][0], weights= sample_probabilities[i][1], k=1)[0])
            
        sample = tuple(sample)
        if sample not in sampled_segments_set and not all([True if s[1] <= 1 else False for s in sample]):
            sampled_segments_set.add(sample)

    return [[list_segments[i][j] for i,j in sample] for sample in sampled_segments_set]

def generate_pairs(curr_iter_segments : list[list[str]], curr_parent_prompts : list[dict], relevant_segments : list[int], N : int) -> list[dict]:

    sampled_segments = sample_segments(curr_iter_segments, relevant_segments, N)

    partions = [p for p in curr_parent_prompts[0]["prompt_partions"]]
    print(partions)
    combined_prompts = []

    for sampled_seg in sampled_segments:
        prompt_dict = {"prompt" : "", "prompt_partions" : [""]*len(partions), "id" : ""}

        for i in range(len(partions)):
            if i not in relevant_segments:
                prompt_dict["prompt_partions"][i] = partions[i]
            else:
                prompt_dict["prompt_partions"][i] = sampled_seg[relevant_segments.index(i)]

        prompt_dict["prompt"] = "<s>[INST]" + "\n\n".join(prompt_dict["prompt_partions"]) + "[/INST]"
        combined_prompts.append(prompt_dict)

        print(prompt_dict["prompt"])

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
    parser.add_argument('--prompts', type=str, help='path to prompts file', default="prompts/EA_Mistral_Prompts_2.json")

    # Output directory
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="outputs/ea_output/")

    # GA parameters
    parser.add_argument('--n_iterations', type=int, help='number of iterations to run GA on', default=5)
    parser.add_argument('--N', type=int, help='number of prompts to sample per iteration', default=20)
    parser.add_argument('--top_k', type=int, help='number of prompts keep for future generations', default=2)
    parser.add_argument('--metric', type=str, help='metric to keep top_k prompts of previous iteration', default="f1_macro")

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", max_length=5000)
    #model.quantize_config = GPTQConfig(bits=4, exllama_config={"version":2}, desc_act=True)
    #model = exllama_set_max_input_length(model, 4096)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, max_length=5000)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset, queries, qrels and prompts
    queries = json.load(open(args.queries))
    qrels = json.load(open(args.qrels))
    prompts = json.load(open(args.prompts))

    # Partion of queries and qrels to use
    iter_choices = random.sample([key for key in queries], k=300)
    iter_queries = {key : queries[key] for key in iter_choices}
    iter_qrels = {key : qrels[key] for key in iter_choices}
    
    relevant_prompt_segments = [0, 1, 4, 6]

    curr_parent_prompts = [{key : prompt[key] for key in prompt if key in ["prompt", "metrics", "prompt_partions", "id"]} for prompt in prompts["scores_f1_macro"][:args.top_k]]

    # Eval in partion set TODO: Uncomment
    #for prompt in curr_parent_prompts:
    #    prompt["metrics"] = GA_evaluation.full_evaluate_prompt(model, tokenizer, iter_queries, iter_qrels, prompt["id"], prompt["prompt"], args, #used_set)

    for i in tqdm(range(1, int(args.n_iterations)+1)):
        # Combine current prompts, generating new prompts
        combination_prompts = combine_curr_prompts(model, tokenizer, prompts["combine_prompt"], relevant_prompt_segments, curr_parent_prompts)

        # Mutate current prompts, generating top_k new prompts
        mutation_prompts = mutate_parent_prompts(model, tokenizer, prompts["mutate_prompt"], relevant_prompt_segments, curr_parent_prompts)

        # Generate all possible combinations
        curr_prompts = generate_pairs([combination_prompts[i] + mutation_prompts[i] for i in range(len(combination_prompts))], curr_parent_prompts, relevant_prompt_segments, args.N)

        for prompt in tqdm(curr_prompts):
            prompt["metrics"] = GA_evaluation.full_evaluate_prompt(model, tokenizer, iter_queries, iter_qrels, prompt["id"], prompt["prompt"], args, used_set)
        
        # Sort curr_prompts by score, comparing to curr_parent_prompts
        curr_prompts = sorted(curr_prompts + curr_parent_prompts, key=lambda x: x["metrics"][args.metric], reverse=True)
        # Cut to top_k prompts
        curr_parent_prompts = curr_prompts[:args.top_k]

        # Output iter res to file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        with safe_open_w(f'{args.output_dir}{timestamp}_EA-Mistral_iter-{i}.json') as f:
            json.dump(curr_prompts, f, indent=4)
        
if __name__ == '__main__':
    main()