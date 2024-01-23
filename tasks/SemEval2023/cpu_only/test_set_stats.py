import os
import json
import argparse
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
    parser.add_argument('--output', type=str, help='path to output dir/file', default='./stats_output.json')
    args = parser.parse_args() 

    queries = json.load(open(f'{args.queries}'))
    output_dict = {}

    for q_id in tqdm(queries):
        key = ""
        if queries[q_id]['Type'] == 'Single':
            key = queries[q_id]['Primary_id']
        elif queries[q_id]['Type'] == 'Comparison':
            key = f'{queries[q_id]["Primary_id"]} + {queries[q_id]["Secondary_id"]}'

        if key not in output_dict:
            output_dict[key] = {"count" : 0, "statements" : []}
        output_dict[key]["count"] += 1
        output_dict[key]["statements"].append(queries[q_id]['Statement'])

    with safe_open_w(args.output) as f:
        json.dump(output_dict, f, indent=4)