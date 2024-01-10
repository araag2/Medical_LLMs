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
    parser.add_argument('--qrels', type=str, help='path to qrels folder', default='../qrels/')
    parser.add_argument('--output', type=str, help='path to output dir/file', default='./stats_output.json')
    args = parser.parse_args() 

    output_dict = {}
    for file in os.listdir(args.qrels):
        qrels = json.load(open(f'{args.qrels}/{file}'))

        single = 0
        comparison = 0
        entailment = 0
        contradiction = 0
        for query in tqdm(qrels):
            if qrels[query]['Type'] == 'Single':
                single += 1
            elif qrels[query]['Type'] == 'Comparison':
                comparison += 1
            
        output_dict[file] = { 'single': single, 'comparison': comparison, 'entailment': entailment, 'contradiction': contradiction }

    with safe_open_w(args.output) as f:
        json.dump(output_dict, f, indent=4)