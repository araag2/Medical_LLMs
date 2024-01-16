import os
import json
import argparse
from tqdm import tqdm

from datetime import datetime

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

    model = "Upstage/SOLAR-10.7B-Instruct-v1.0"

    sorted_res = {"model" : model, "set" : "dev", "scores_f1_macro" : []}
    json_files = [json_f for json_f in os.listdir(args.sort_folder) if json_f.endswith('.json')]

    for json_f in tqdm(json_files):
        with open(args.sort_folder + json_f, 'r') as f:
            data = json.load(f)
            if data["model"] == model:
                sorted_res["scores_f1_macro"].append({key : data[key] for key in data if key in ["prompt", "metrics", "formated_metrics"]})
                sorted_res["scores_f1_macro"][-1]["id"] = sorted_res["scores_f1_macro"][-1]["formated_metrics"].split('|')[1].rstrip()[-7:]
                sorted_res["scores_f1_macro"][-1]["prompt_partions"] = sorted_res["scores_f1_macro"][-1]["prompt"].split("\n\n")
    
    sorted_res["scores_f1_macro"] = sorted(sorted_res["scores_f1_macro"], key=lambda x: x["metrics"]["f1_macro"], reverse=True)[:20]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H")

    with safe_open_w(f'{args.output_dir}EA_base_prompts-{timestamp}.json') as f:
        json.dump(sorted_res, f, indent=4)

if __name__ == '__main__':
    main()