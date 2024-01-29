import os
import argparse
import json
import torch
import GA_evaluation

from transformers import GPTQConfig, LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model

#if "SLURM_JOB_ID" not in os.environ:
#    device = "CPU"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='name of the model used to generate and combine prompts', default='mistralai/Mistral-7B-Instruct-v0.2')

    used_set = "test" # train | dev | test

    # Path to queries, qrels and prompt files
    parser.add_argument('--queries', type=str, help='path to queries file', default=f'queries/queries2024_{used_set}.json')
    parser.add_argument('--qrels', type=str, help='path to qrels file', default=f'qrels/qrels2024_{used_set}.json')
    # "prompts/T5prompts.json"
    parser.add_argument('--prompts', type=str, help='path to prompts file', default="prompts/EA_Mistral_Prompts_2.json")
    parser.add_argument('--checkpoint', type=str, help='path to prompts file', default="outputs/models/run_11/end_model/")

    # Output directory
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="outputs/test_set_results/")

    args = parser.parse_args()

    base_model_reload = AutoModelForCausalLM.from_pretrained(
       args.model, low_cpu_mem_usage=True,
       return_dict=True,torch_dtype=torch.bfloat16,
       device_map= {"": 0}
    )
    #new_model = AutoModelForCausalLM.from_pretrained(f'outputs/models/run_2/checkpoint-2125/')
    model = PeftModel.from_pretrained(base_model_reload, args.checkpoint)
    model = model.merge_and_unload()

    model.save_pretrained("outputs/models/run_11/end_model/")
    quit()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset, queries, qrels and prompts
    queries = json.load(open(args.queries))
    qrels = json.load(open(args.qrels))
    #prompts = json.load(open(args.prompts))

    prompt = "<s>[INST]The objective is to examine semantic entailment relationships between individual sections of Clinical Trial Reports (CTRs) and statements articulated by clinical domain experts. CTRs elaborate on the procedures and findings of clinical trials, scrutinizing the effectiveness and safety of novel treatments. Each trial involves cohorts or arms exposed to distinct treatments or exhibiting diverse baseline characteristics. Comprehensive CTRs comprise four sections: (1) ELIGIBILITY CRITERIA delineating conditions for patient inclusion, (2) INTERVENTION particulars specifying type, dosage, frequency, and duration of treatments, (3) RESULTS summary encompassing participant statistics, outcome measures, units, and conclusions, and (4) ADVERSE EVENTS cataloging signs and symptoms observed. Statements posit claims regarding the information within these sections, either for a single CTR or in comparative analysis of two. To establish entailment, the statement's assertion should harmonize with clinical trial data, find substantiation in the CTR, and avoid contradiction with the provided descriptions.\n\nThe following descriptions correspond to the information in one of the Clinical Trial Report (CTR) sections.\n\nPrimary Trial:\n$primary_evidence\n\nSecondary Trial:\n$secondary_evidence\n\nReflect upon the ensuing statement crafted by an expert in clinical trials.\n\n$hypothesis\n\nRespond with either YES or NO to indicate whether it is possible to determine the statement's validity based on the Clinical Trial Report (CTR) information, with the statement being supported by the CTR data and not contradicting the provided descriptions.[/INST] Answer: "

    GA_evaluation.output_prompt_labels(model, tokenizer, queries, prompt, args, used_set)

    #GA_evaluation.full_evaluate_prompt(model, tokenizer, queries, qrels, "best_gen_prompt", prompt, args, used_set)

if __name__ == '__main__':
    main()