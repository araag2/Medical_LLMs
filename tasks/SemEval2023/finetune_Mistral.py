import sys, os
import wandb
import json
import pandas as pd
import torch
import argparse
import loralib as lora

#Local Files
from GA_evaluation import create_qid_prompt_label_dict

from dataclasses import dataclass, field
from datasets.arrow_dataset import Dataset
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType 
from subprocess import call
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig,AutoPeftModelForCausalLM
from trl import SFTTrainer
from typing import List, Type, Optional
from tqdm import tqdm
from datasets.arrow_dataset import Dataset
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def preprocess_dataset(args : argparse, tokenizer : AutoTokenizer, prompt : str , split : str):
    # Load JSON
    set_examples = create_qid_prompt_label_dict(json.load(open(f'{args.queries}queries2024_{split}.json')), json.load(open(f'{args.qrels}qrels2024_{split}.json')), prompt)
    
    set_dict = {"id" : [], "input_ids" : [], "token_type_ids" : [], "attention_mask" : [], "label" : []}
    print(set_examples)
    for q_id in set_examples:
        example = set_examples[q_id]
        tokenized = tokenizer(example["text"], return_tensors="pt")

        set_dict["id"].append(q_id)
        set_dict["input_ids"].append(tokenized["input_ids"].squeeze())
        set_dict["token_type_ids"].append(tokenized["token_type_ids"].squeeze())
        set_dict["attention_mask"].append(tokenized["attention_mask"].squeeze())
        set_dict["label"].append(example["label"])
    return Dataset.from_dict(set_dict)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help='model to train')
    parser.add_argument('--exp_name', type=str, default="Mistral SemEval Fine-Tune", help='Describes the conducted experiment')
    parser.add_argument('--run', type=int, default=0, help='run number for wandb logging')

    # I/O paths for models, CT, queries and qrels
    #parser.add_argument('--load_dir', type=str, default="LMHead/", help='path to model load dir')
    parser.add_argument('--save_dir', type=str, default="outputs/models/", help='path to model save dir')
    #parser.add_argument("--CT_input", default="datasets/TREC2021/TREC2021_CT_corpus.json", type=str, help='path to JSON for MLM')

    parser.add_argument("--used_prompt", default="prompts/", type=str)
    parser.add_argument("--queries", default="queries/", type=str)
    parser.add_argument("--qrels", default="qrels/", type=str)

    #Model Hyperparamenters
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--train_epochs", default=5, type=int)
    parser.add_argument("--lr", type=float, default=2e-5)

    # Lora Hyperparameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_alpha", type=float, default=0.1)

    #Speed and memory optimization parameters
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision instead of 32-bit")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="If True, use gradient checkpointing to save memory at the expense of slower backward pass.")
    args = parser.parse_args()


    return args

def create_model_and_tokenizer(args : argparse):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.bfloat16,
        bnb_4bit_use_double_quant= False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config= bnb_config,
        device_map= {"": 0},
        # use_auth_token=True,
        # revision="refs/pr/35"
    )

    #### LLAMA STUFF
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r = args.lora_r,
        lora_alpha= args.lora_alpha,
        lora_dropout= args.lora_dropout,
        # target_modules=["query_key_value"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj"],
    )

    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer

def main():
    args = parse_args()

    #wandb.init(
    #    project="SemEval_Mistra",
    #    name = f'{args.model_name}/{args.exp_name}/run-{args.run}',
    #    group = f'{args.model_name}/{args.exp_name}',
    #    config = { arg : getattr(args, arg) for arg in vars(args)}
    #)

    # Load tokenizer and model
    model, peft_config, tokenizer = create_model_and_tokenizer(args)

    #TODO: Load the prompt correctly
    prompt = "The objective is to examine semantic entailment relationships between individual sections of Clinical Trial Reports (CTRs) and statements articulated by clinical domain experts. CTRs elaborate on the procedures and findings of clinical trials, scrutinizing the effectiveness and safety of novel treatments. Each trial involves cohorts or arms exposed to distinct treatments or exhibiting diverse baseline characteristics. Comprehensive CTRs comprise four sections: (1) ELIGIBILITY CRITERIA delineating conditions for patient inclusion, (2) INTERVENTION particulars specifying type, dosage, frequency, and duration of treatments, (3) RESULTS summary encompassing participant statistics, outcome measures, units, and conclusions, and (4) ADVERSE EVENTS cataloging signs and symptoms observed. Statements posit claims regarding the information within these sections, either for a single CTR or in comparative analysis of two. To establish entailment, the statement's assertion should harmonize with clinical trial data, find substantiation in the CTR, and avoid contradiction with the provided descriptions.\n\nThe following descriptions correspond to the information in one of the Clinical Trial Report (CTR) sections.\n\nPrimary Trial:\n$primary_evidence\n\nSecondary Trial:\n$secondary_evidence\n\nReflect upon the ensuing statement crafted by an expert in clinical trials.\n\n$hypothesis\n\nRespond with either YES or NO to indicate whether it is possible to determine the statement's validity based on the Clinical Trial Report (CTR) information, with the statement being supported by the CTR data and not contradicting the provided descriptions. "

    # Load dataset
    train_dataset = preprocess_dataset(args, tokenizer, prompt, "train")
    eval_dataset = preprocess_dataset(args, tokenizer, prompt, "test")

    print(train_dataset)
    print(eval_dataset)
    print(model)

    #training_arguments = TrainingArguments(
    #    output_dir = args.save_dir,
    #    overwrite_output_dir=True,
    #    num_train_epochs = args.train_epochs,
    #    per_device_train_batch_size= args.batch_size,
    #    optim = "paged_adamw_8bit",
    #    save_steps= 1000,
    #    save_total_limit= 4,
    #    logging_steps= 100,
    #    learning_rate= args.lr,
    #    weight_decay= 0.001,
    #    bf16= False,
    #    #max_grad_norm= 0.3,
    #    #max_steps= -1,
    #    #warmup_ratio= 0.3,
    #    group_by_length= True,
    #    lr_scheduler_type= "constant",
    #
    #    #Speed and memory optimization parameters
    #    gradient_accumulation_steps= args.gradient_accumulation_steps,
    #    gradient_checkpointing= args.gradient_checkpointing,
    #    fp16= args.fp16,
    #    report_to="wandb"
    #)
    #
    ## Setting sft parameters
    #trainer = SFTTrainer(
    #    model= model,
    #    train_dataset= dataset,
    #    eval_dataset= dataset,
    #    peft_config= peft_config,
    #    max_seq_length= 4096,
    #    dataset_text_field= "chat_sample",
    #    tokenizer= tokenizer,
    #    args= training_arguments,
    #    packing= False,
    #)
    #
    ## Training
    #trainer.train()
    #trainer.model.save_pretrained(args.save_dir)



if __name__ == '__main__':
    main()