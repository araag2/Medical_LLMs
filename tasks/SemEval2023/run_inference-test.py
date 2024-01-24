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
    parser.add_argument('--checkpoint', type=str, help='path to prompts file', default="outputs/models/run_6/checkpoint-2120/")

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

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset, queries, qrels and prompts
    queries = json.load(open(args.queries))
    qrels = json.load(open(args.qrels))
    #prompts = json.load(open(args.prompts))

    prompt = "<s>[INST]Consider the task of determining semantic entailment relations between individual sections of Clinical Trial Reports (CTRs) and statements made by clinical domain experts. Note that CTRs outline the methodology and findings of a clinical trial, which are conducted to assess the effectiveness and safety of new treatments. Each trial involves 1-2 patient groups, called cohorts or arms, and these groups may receive different treatments, or have different baseline characteristics. The complete CTRs contain 4 sections, corresponding to (1) a list of the ELIGIBILITY CRITERIA corresponding to the conditions for patients to be allowed to take part in the clinical trial, (2) a description for the INTERVENTION that specifies the type, dosage, frequency, and duration of treatments being studied, (3) a summary of the RESULTS, detailing aspects such as the number of participants in the trial, the outcome measures, the units, and the conclusions, and (4) a list of ADVERSE EVENTS corresponding to signs and symptoms observed in patients during the clinical trial. In turn, the statements are sentences that make some type of claim about the information contained in one of the aforementioned sections, either considering a single CTR or comparing two CTRs. In order for the entailment relationship to be established, the claim in the statement should be related to the clinical trial information, it should be supported by the CTR, and it must not contradict the provided descriptions.\n\nThe given and provided descriptions align with the information in a particular section of Clinical Trial Reports (CTRs), detailing relevant and matching content. \n\nPrimary Trial:\n$primary_evidence\n\nSecondary Trial:\n$secondary_evidence\n\nClinical domain experts, clinical trial organizers, or medical researchers may propose the following statement. \n\n$hypothesis\n\nBased on the clinical trial report, can you determine if the statement is valid? (Answer with 'Yes' or 'No')[/INST] Answer: "

    GA_evaluation.output_prompt_labels(model, tokenizer, queries, prompt, args, used_set)

    #GA_evaluation.full_evaluate_prompt(model, tokenizer, queries, qrels, "best_gen_prompt", prompt, args, used_set)

if __name__ == '__main__':
    main()