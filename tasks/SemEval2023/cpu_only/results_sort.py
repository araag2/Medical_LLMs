import os
import json
import argparse
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = "CPU"

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def main():
    parser = argparse.ArgumentParser()

    # "prompts/T5prompts.json"
    parser.add_argument('--sort_folder', type=str, help='path to prompts file', default="../outputs/combination_output/")
    # Output directory
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="../outputs/")
    args = parser.parse_args()

    sorted_res = {"model" : "mistralai/Mistral-7B-Instruct-v0.2", "set" : "dev", "scores_precision" : []}
    json_files = [json_f for json_f in os.listdir(args.sort_folder) if json_f.endswith('.json')]

    for json_f in tqdm(json_files):
        with open(args.sort_folder + json_f, 'r') as f:
            data = json.load(f)
            sorted_res["scores_precision"].append({key : data[key] for key in data if key in ["prompt", "metrics", "formated_metrics"]})
    
    sorted_res["scores_precision"].sort(key=lambda x: x["metrics"]["precision"], reverse=True)
    sorted_res["scores_f1"] = sorted(sorted_res["scores_precision"], key=lambda x: x["metrics"]["f1"], reverse=True)

    with safe_open_w(f'{args.output_dir}sorted_base_prompts.json') as f:
        json.dump(sorted_res, f, indent=4)

if __name__ == '__main__':
    main()