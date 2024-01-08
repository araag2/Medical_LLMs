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
    parser.add_argument('--input', type=str, help='path to input dir/file', required=True)
    parser.add_argument('--output', type=str, help='path to output dir/file', required=True)
    parser.add_argument('--type', type=str, help='type: trials, queries or qrels', required=True, default='trials')
    args = parser.parse_args()

    if args.type not in ["trials", "queries", "qrels"]:
        print('Argument value should be "trials" or "queries" or "qrels"')
        exit()

    section_substitution = {"Intervention" : "interventions",
                            "Eligibility" : "eligibity criteria",
                            "Adverse Events" : "adverse events",
                            "Results" : "results"}
    #relevant_qrels_info = ALL      

    with safe_open_w(args.output) as output_file:
        output_dict = {}

        # Handling of CT (Clinical Trials) files
        if args.type == 'trials':
            for root, dirs, files in os.walk(args.input):
                files.remove(".DS_Store")
                CT_corpus = {}
                for file in tqdm(files):
                    with open(os.path.join(os.getcwd(), root, file), 'r') as current_CT:
                        json_CT = json.load(current_CT)
                        
                        CT_corpus[json_CT["Clinical Trial ID"]] = {}
                        for data_field in json_CT:
                            if data_field != "Clinical Trial ID": 
                                CT_corpus[json_CT["Clinical Trial ID"]][data_field] = json_CT[data_field]

                output_dict = CT_Corpus

        # Handling of query files
        elif args.type == 'queries':
            output_queries = {}
            with open(args.input) as input_queries_f:
                json_input_queries = json.load(input_queries_f)

                for query in tqdm(json_input_queries):  
                    output_queries[query] = {}
                    for data_field in ['Type', 'Primary_id', 'Secondary_id', 'Section_id']:
                        if data_field in json_input_queries[query]:
                            output_queries[query][data_field] = json_input_queries[query][data_field]
                   
                    #Uncapitalize Statement and remove "."
                    sentence = json_input_queries[query]['Statement']
                    sentence = f'{sentence[0].lower()}{sentence[1:]}'
                    sentence = sentence if sentence[-1] != "." else sentence[:-1].rstrip()
                    output_queries[query]['Statement'] = sentence

            output_dict = output_queries

            #Generates extended queries, that include origin CT segments, in order to create actual prompts
            with safe_open_w(f'{args.output[:-5]}_extended.json') as output_file_extended:
                extended_query_dict = copy.deepcopy(output_queries)
                with open(f'{os.getcwd()}/data_json/SemEval2023/CT_corpus.json') as JSON_Corpus:
                    corpus = json.load(JSON_Corpus)
                    
                    for query in extended_query_dict:
                        extended_query_dict[query]['Primary_id_txt_list'] = corpus[extended_query_dict[query]['Primary_id']][extended_query_dict[query]['Section_id']]
                        extended_query_dict[query]['Primary_id_txt'] = "\n".join(line for line in extended_query_dict[query]['Primary_id_txt_list'])
                        if 'Secondary_id' in extended_query_dict[query]:
                            extended_query_dict[query]['Secondary_id_txt_list'] = corpus[extended_query_dict[query]['Secondary_id']][extended_query_dict[query]['Section_id']]
                            extended_query_dict[query]['Secondary_id_txt'] = "\n".join(line for line in extended_query_dict[query]['Secondary_id_txt_list'])

                for query in extended_query_dict:    
                    # Change section to natural language
                    output_queries[query]['Section_id'] = section_substitution[json_input_queries[query]['Section_id']]
                    extended_query_dict[query]['Section_id'] = section_substitution[json_input_queries[query]['Section_id']]
                output_file_extended.write(json.dumps(extended_query_dict, ensure_ascii=False, indent=4))


        # Handling of qrels files
        elif args.type == 'qrels':
            output_qrels = {}
            with open(args.input) as input_qrels_f:
                json_input_qrels = json.load(input_qrels_f)

                for qrel in tqdm(json_input_qrels):
                    output_qrels[qrel] = {}
                    for data_field in json_input_qrels[qrel]:
                        output_qrels[qrel][data_field] = json_input_qrels[qrel][data_field]

            output_dict = output_qrels

        # Output to file
        output_file.write(json.dumps(output_dict, ensure_ascii=False, indent=4))
                    