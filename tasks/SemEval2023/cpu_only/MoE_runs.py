import os
import json
import argparse
import random
from tqdm import tqdm
from typing import List

def safe_open_w(path: str):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=str, help='path to qrels folder', default='queries/queries2024_test.json')
    parser.add_argument('--ts_1', type=str, help='path to test_set_results folder', default='outputs/test_set_results/2024-01-23_20-08_BEST-SCORE_run_2-checkpoint-1275.json')
    parser.add_argument('--ts_2', type=str, help='path to test_set_results folder', default='outputs/test_set_results/2024-01-28_16-09_run_7_checkpoint-12388_test-set.json')
    parser.add_argument('--ts_3', type=str, help='path to test_set_results folder', default='outputs/test_set_results/2024-01-26_15-04_run_8_end_model_test-set.json')
    parser.add_argument('--ts_4', type=str, help='path to test_set_results folder', default='outputs/test_set_results/2024-01-29_20-09_run_11_end_model_test-set.json')
   
    parser.add_argument('--output', type=str, help='path to output dir/file', default='./cpu_only/res_output.json')
    args = parser.parse_args() 

    queries = json.load(open(f'{args.queries}'))
    ts1 = json.load(open(f'{args.ts_1}'))
    ts2 = json.load(open(f'{args.ts_2}'))
    ts3 = json.load(open(f'{args.ts_3}'))
    ts4 = json.load(open(f'{args.ts_4}'))


    #for q_id in tqdm(queries):
    #    key = ""
    #    if queries[q_id]['Type'] == 'Single':
    #        key = queries[q_id]['Primary_id']
    #    elif queries[q_id]['Type'] == 'Comparison':
    #        key = f'{queries[q_id]["Primary_id"]} + {queries[q_id]["Secondary_id"]}'
    #
    #    if key not in output_dict:
    #        output_dict[key] = {"count" : 0, "statements" : []}
    #    output_dict[key]["count"] += 1
    #    output_dict[key]["statements"].append((queries[q_id]['Statement']))
    #
    #with safe_open_w(args.output) as f:
    #    json.dump(output_dict, f, indent=4)


        
    output_dict = {}
    n_total = len(ts1)
    n_all_negs = 0
    n_all_pos = 0
    for key in ts1:
        labels = [ts1[key]['Prediction'], ts2[key]['Prediction'], ts3[key]['Prediction'], ts4[key]['Prediction']]
        n_neg = labels.count("Contradiction")
        n_pos = labels.count("Entailment")
        if n_neg >= n_pos:
            output_dict[key] = {"Prediction" : "Contradiction"}
        else:
            output_dict[key] = {"Prediction" : "Entailment"}

    with safe_open_w(args.output) as f:
        json.dump(output_dict, f, indent=4)