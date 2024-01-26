import os
import json
import string
import argparse
import copy
import pandas as pd
import xml.etree.ElementTree as ET

from collections import OrderedDict
from tqdm import tqdm
from typing import List

def safe_open_w(path: str):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to input dir/file', default='qrels/qrels2024_train-dev_manual-Expand-2.json')
    parser.add_argument('--output', type=str, help='path to output dir/file', default='queries/queries2024_train-dev_manual-Expand-2.json')
    parser.add_argument('--corpus', type=str, help='path to CT Corpus', default='CT_json/SemEval_CT-corpus.json')
    args = parser.parse_args() 

    qrels = json.load(open(args.input))
    corpus = json.load(open(args.corpus, encoding="utf8"))

    output_dict = {}

    #TO:DO 
    for query in tqdm(qrels):  
        output_dict[query] = {}
        for data_field in ['Type', 'Section_id', 'Primary_id', 'Secondary_id', 'Section_id', 'Statement']:
            if data_field in qrels[query]:
               output_dict[query][data_field] = qrels[query][data_field]
                    
        output_dict[query]['Primary_id_txt_list'] = corpus[output_dict[query]['Primary_id']][output_dict[query]['Section_id']]
        output_dict[query]['Primary_id_txt'] = "\n".join(line for line in output_dict[query]['Primary_id_txt_list'])
        if 'Secondary_id' in output_dict[query]:
            output_dict[query]['Secondary_id_txt_list'] = corpus[output_dict[query]['Secondary_id']][output_dict[query]['Section_id']]
            output_dict[query]['Secondary_id_txt'] = "\n".join(line for line in output_dict[query]['Secondary_id_txt_list'])

    with safe_open_w(args.output) as f:
        json.dump(output_dict, f, indent=4)