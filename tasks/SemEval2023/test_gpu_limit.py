import argparse
import torch
import os

#from auto_gptq import exllama_set_max_input_length
from tqdm import tqdm
from transformers import GPTQConfig, LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='name of the T5 model used', default='Upstage/SOLAR-10.7B-Instruct-v1.0')
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    #model = LlamaForCausalLM.from_pretrained(args.model_name, device_map="auto")
    #model = AutoModelForCausalLM.from_pretrained("TheBloke/qCammel-70-x-GPTQ",
    #                                            torch_dtype=torch.float16,
    #                                            device_map="auto",
    #                                            revision="gptq-4bit-128g-actorder_True")
    #model.quantize_config = GPTQConfig(bits=4, exllama_config={"version":2}, desc_act=True)
    #model = exllama_set_max_input_length(model, 4096)
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)

    print(f'Tokenized {args.model_name=} tokens')

    with torch.inference_mode():
        for i in tqdm(range(2000, 50000, 200)):
            text = " ".join(["average"] * i)

            input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
            print(f'Tokenized {input_ids.shape=} tokens')

            outputs = model.generate(input_ids, max_new_tokens=100)
            print(f'Generated {outputs[0].shape=} tokens')

            if i == 5000:
                quit()
            
if __name__ == '__main__':
    main()