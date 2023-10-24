# Medical_LLMs
Repository to act as a playground to test various LLMs in the context of the medical domain 

## SemEval2023

### Set Compostions 

All sets are balanced in labels, having 50% of Entailment and Contradiction

| **Metrics**    | #Samples | Single | Comparison |
|:-------------- |:--:|:--:|:--:|
| Train | 1700 | 1035 | 665 |
| Dev   | 200  | 140  | 60  | 
| Test  | 500  | 229  | 271 |

---

### T5 Results

In order to obtain these results, I used [Flan-T5 from huggingface](https://huggingface.co/google/flan-t5-base), and used the following generation prompt: `$premise \n Question: Does this imply that $hypothesis? $options`, checking the outputs for "Entailment" or "Contradiction".

#### Train Set (0-shot)

| **Metrics**    | F1-score | Precision | Recall | Notes |
|:-------------- |:--:|:--:|:--:|:--:|
| flanT5-small | - | - | - | Always Contradiction |
| flanT5-base | 0.32 | 0.50 | 0.23 | - |
| flanT5-large | 0.53 | 0.56 | 0.49 | - |
| flanT5-xl | 0.67 | 0.59 | 0.77 | - |
| flanT5-xxl | 0.69 | 0.61 | 0.79 | - |

#### Dev Set (0-shot)

| **Metrics**    | F1-score | Precision | Recall | Notes |
|:-------------- |:--:|:--:|:--:|:--:|
| flanT5-small | - | - | - | Always Contradiction |
| flanT5-base | 0.34 | 0.55 | 0.25 | - |
| flanT5-large | 0.57 | 0.61 | 0.53 | - |
| flanT5-xl | 0.69 | 0.61 | 0.79 | - |
| flanT5-xxl | 0.71 | 0.59 | 0.88 | - |

#### Dev Set (fine-tuned)

| **Metrics**    | F1-score | Precision | Recall | Notes |
|:-------------- |:--:|:--:|:--:|:--:|
| flanT5-xl | 0.754 | 0.59 | 0.831 | - |

---

### qCammel family Results

In order to use the Clinical Cammel models, we will obtain them from [huggingface hub](https://huggingface.co/TheBloke), as it very easily allows us to quantize and test different model sizes. 

#### Prompts

Alpaca Template (_alp{prompt_used}): `Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n$task_prompt\n### Response:`

Asclepius Template (_asclp{prompt_used}): `You are an intelligent clinical languge model.\nBelow is a snippet of clinical trial data and a following instruction from a healthcare professional. Write a response that appropriately completes the instruction. The response should provide the accurate answer to the instruction, while being concise.\n[Discharge Summary Begin]\nNotes go here\n[Discharge Summary End]\n\n[Instruction Begin]\n{prompt}\n[Instruction End]`

Base T5 Prompt (t5): `$premise \n Question: Does this imply that $hypothesis? $options`

#### Perplexity

Instead of trying to generate "Entailment" and "Contradiction", it's also possible to analize the output probabilities of the tokens, not requiring the model to explicity generate those tokens. 

We will denote these differences by using `_gen` when the full generation is processed, and `_perp` when the perplexity score is used.

#### Model Links
[Asclepius-Llama2-13B](https://huggingface.co/starmpcc/Asclepius-Llama2-13B)
[Asclepius-13B-GPTQ](https://huggingface.co/TheBloke/Asclepius-13B-GPTQ)

[qCammel-13-GPTQ](https://huggingface.co/TheBloke/qCammel-13-GPTQ)
[Cammel-13B-Combined-Data-GPTQ](https://huggingface.co/TheBloke/CAMEL-13B-Combined-Data-GPTQ)
[qCammel-70-x-GPTQ](https://huggingface.co/TheBloke/qCammel-70-x-GPTQ)
[qCammel-70-x-GPTQ-gptq-3bit-128g](https://huggingface.co/TheBloke/qCammel-70-x-GPTQ/tree/gptq-3bit-128g-actorder_True)

---

#### Train Set (0-shot)


| **Metrics**    | F1-score | Precision | Recall | Notes |
|:-------------- |:--:|:--:|:--:|:--:|
| qCammel-13-GPTQ_(t5)_(gen) | 0.64 | 0.60 | 0.69 | - |
| Asclepius-Llama2-13B_(t5)_(gen) | 0.62 | 0.59 | 0.64 | - |

#### Dev Set (0-shot)

| **Metrics**    | F1-score | Precision | Recall | Notes |
|:-------------- |:--:|:--:|:--:|:--:|
| qCammel-13-GPTQ_(t5)_(gen) | 0.65 | 0.62 | 0.68 | - |
| Asclepius-Llama2-13B_(t5)_(gen) | 0.61 | 0.59 | 0.63 | - |

---