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

Alpaca Template (_alp + {prompt_used}): `Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n$task_prompt\n### Response:`

Asclepius Template (_asclp + {prompt_used}): `You are an intelligent clinical language model.\nBelow is a snippet of clinical trial data and a following instruction from a healthcare professional. Write a response that appropriately completes the instruction. The response should provide the accurate answer to the instruction, while being concise.\n\n[Instruction Begin]\n{prompt}\n[Instruction End]`

Base T5 Prompt (t5): `$premise \n Question: Does this imply that $hypothesis? $options`

#### Perplexity

Instead of trying to generate "Entailment" and "Contradiction", it's also possible to analize the output probabilities of the tokens, not requiring the model to explicity generate those tokens. 

We will denote these differences by using `_gen` when the full generation is processed, and `_perp` when the perplexity score is used.

#### Model Links
[Asclepius-Llama2-13B](https://huggingface.co/starmpcc/Asclepius-Llama2-13B)

[Asclepius-13B-GPTQ](https://huggingface.co/TheBloke/Asclepius-13B-GPTQ)

[qCammel-13-GPTQ](https://huggingface.co/TheBloke/qCammel-13-GPTQ)

[qCammel-13B-Combined-Data-GPTQ](https://huggingface.co/TheBloke/CAMEL-13B-Combined-Data-GPTQ)

[qCammel-13B-Role-Playing-GPTQ](https://huggingface.co/TheBloke/CAMEL-13B-Role-Playing-Data-GPTQ)

[qCammel-70-x-GPTQ](https://huggingface.co/TheBloke/qCammel-70-x-GPTQ)

[qCammel-70-x-GPTQ-gptq-4bit-128g](https://huggingface.co/TheBloke/qCammel-70-x-GPTQ/tree/gptq-4bit-128g-actorder_True)

[qCammel-70-x-GPTQ-gptq-3bit-128g](https://huggingface.co/TheBloke/qCammel-70-x-GPTQ/tree/gptq-3bit-128g-actorder_True)

---

#### Dev Set (0-shot)

##### Testing different task prompts (with apl{t5} + gen)

`"1" : "$premise \n Based on the paragraph above can we conclude that $hypothesis? $options",`

`"2" : "$premise \n Based on that paragraph can we conclude that this sentence is true? $hypothesis $options",`

`"3" : "$premise \n Can we draw the following conclusion? $hypothesis $options",`

`"4" : "$premise \n Does this next sentence follow, given the preceding text? $hypothesis $options",`

`"5" : "$premise \n Can we infer the following? $hypothesis $options",`

`"6" : "Read the following paragraph and determine if the hypothesis is true: $premise \n Hypothesis: $hypothesis \n $options",`

`"7" : "Read the text and determine if the sentence is true: $premise \n Hypothesis: $hypothesis \n $options",`

`"8" : "Can we draw the following hypothesis from the context? Context: $premise \n Hypothesis: $hypothesis \n $options",`

`"9" : "Determine if the sentence is true based on the text below: $hypothesis \n $premise \n  $options",`

`"10" : "$premise \n Question: Does this imply that $hypothesis? $options"`

| **Metrics**    | F1-score | Precision | Recall | Notes |
|:-------------- |:--:|:--:|:--:|:--:|
| qCammel-13-GPTQ_(apl{t5})_(gen)_1  | 0.57 | 0.57 | 0.57 | - |
| qCammel-13-GPTQ_(apl{t5})_(gen)_2  | 0.67 | 0.51 | 0.96 | mostly yes |
| qCammel-13-GPTQ_(apl{t5})_(gen)_3  | 0.65 | 0.63 | 0.85 | - |
| qCammel-13-GPTQ_(apl{t5})_(gen)_4  | 0.66 | 0.51 | 0.94 | mostly yes |
| qCammel-13-GPTQ_(apl{t5})_(gen)_5  | 0.65 | 0.50 | 0.98 | all yes |
| qCammel-13-GPTQ_(apl{t5})_(gen)_6  | 0.66 | 0.5 | 1.0 | all yes |
| qCammel-13-GPTQ_(apl{t5})_(gen)_7  | 0.66 | 0.5 | 1.0 | all yes |
| qCammel-13-GPTQ_(apl{t5})_(gen)_8  | 0.67 | 0.61 | 0.75 | - |
| qCammel-13-GPTQ_(apl{t5})_(gen)_9  | 0.64 | 0.58 | 0.71 | - |
| qCammel-13-GPTQ_(apl{t5})_(gen)_10 | 0.635 | 0.530 | 0.800 | - |

| **Metrics**    | F1-score | Precision | Recall | Notes |
|:-------------- |:--:|:--:|:--:|:--:|
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_1  | 0.71 | 0.63 | 0.82 | x |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_2  | 0.65 | 0.65 | 0.65 | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_3  | 0.70 | 0.61 | 0.83 | x |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_4  | 0.59 | 0.57 | 0.60 | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_5  | 0.70 | 0.56 | 0.94 | x |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_6  | 0.62 | 0.68 | 0.57 | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_7  | 0.60 | 0.64 | 0.57 | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_8  | 0.69 | 0.57 | 0.88 | x |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_9  | 0.55 | 0.53 | 0.58 | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_10 | 0.72 | 0.61 | 0.86 | x |

---

##### Using Alpaca Template (_alp) + T5 prompt yes/no (t5) + generation (gen)

Prompt: `Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n$premise \n Question: Does this imply that $hypothesis?\n Respond by outputting Yes or No. Be as accurate as possible. \n\n\n### Response:`

| **Metrics**    | F1-score | Precision | Recall | Notes |
|:-------------- |:--:|:--:|:--:|:--:|
| Asclepius-Llama2-13B_(apl{t5})_(gen) | 0.55 | 0.54 | 0.56 | - |
| Asclepius-13B-GPTQ_(apl{t5})_(gen) | 0.56 | 0.51 | 0.62 | - |
| qCammel-13-GPTQ_(apl{t5})_(gen) | 0.635 | 0.53 | 0.80 | - |
| qCammel-13-Combined-Data-GPTQ_(apl{t5})_(gen) | 0.57 | 0.525 | 0.63 | - |
| qCammel-13-Role-Playing-GPTQ_(apl{t5})_(gen) | 0.60 | 0.52 | 0.72 | 17 forced yes |
| qCammel-70-x-GPTQ_(apl{t5})_(gen) | 0.67 | 0.68 | 0.65 | had to reduce max len to 2k tokens |
| qCammel-70-x-GPTQ-gptq-3bit-128g_(apl{t5})_(gen) | 0.705 | 0.67 | 0.74 | - |

##### Using Alpaca Template (_alp) + T5 prompt entailment/contradiction (t5) + generation (gen)

Prompt: `Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n$premise \n Question: Does this imply that $hypothesis?\n Respond by outputting Entailment or Contradiction. Be as accurate as possible. \n\n\n### Response:`

| **Metrics**    | F1-score | Precision | Recall | Notes |
|:-------------- |:--:|:--:|:--:|:--:|
| Asclepius-Llama2-13B_(apl{t5})_(gen) | 0.49 | 0.45 | 0.54 | - |
| Asclepius-13B-GPTQ_(apl{t5})_(gen) | 0.49 | 0.41 | 0.61 | - |
| qCammel-13-GPTQ_(apl{t5})_(gen) | 0.632 | 0.53 | 0.79 | - |
| qCammel-13-Combined-Data-GPTQ_(apl{t5})_(gen) | 0.48 | 0.49 | 0.48 | - |
| qCammel-13-Role-Playing-GPTQ_(apl{t5})_(gen) | 0.52 | 0.51 | 0.54 | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen) | 0.66 | 0.50 | 0.96 | 192/200 Entailment |
| qCammel-70-x-GPTQ-gptq-3bit-128g_(apl{t5})_(gen) | 0.67 | 0.505 | 1 | 198/200 Entailment |

---

##### Using Alpaca Template (_alp) + T5 prompt yes/no (t5) + generation with CoT (gen-CoT)

Prompt: `Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n$premise \n Question: Does this imply that $hypothesis?\n Let's think step by step, and end by outputting Yes or No. Be as accurate as possible. \n\n\n### Response:`

| **Metrics**    | F1-score | Precision | Recall | Notes |
|:-------------- |:--:|:--:|:--:|:--:|
| qCammel-13-GPTQ_(apl{t5})_(gen) | - | - | - | - |
| qCammel-13-Combined-Data-GPTQ_(apl{t5})_(gen) | - | - | - | - |
| qCammel-13-Role-Playing-GPTQ_(apl{t5})_(gen) | - | - | - | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen) | - | - | - | had to reduce max len to 2k tokens |
| qCammel-70-x-GPTQ-gptq-3bit-128g_(apl{t5})_(gen) | - | - | - | - |

---

#### New Prompts Generation
`"1" : "$premise \n Based on the paragraph above can we conclude that $hypothesis? $options",`
`"3" : "$premise \n Can we draw the following conclusion? $hypothesis $options",`
`"5" : "$premise \n Can we infer the following? $hypothesis $options",`
`"8" : "Can we draw the following hypothesis from the context? Context: $premise \n Hypothesis: $hypothesis \n $options",`
`"10" : "$premise \n Question: Does this imply that $hypothesis? $options"`

Using these base prompts and gpt3.5-turbo, I generated the following prompts:
`"1_3"  : "$premise Can we draw a reasonable conclusion regarding $hypothesis? Consider the $options",`
`"1_5"  : "$premise Can we draw a conclusion regarding $hypothesis based on the provided information? Consider the $options",`
`"1_8"  : "$premise Can we derive a conclusion regarding $hypothesis from the information provided? Consider the $options",`
`"1_10" : "$premise \n Given the context, is it reasonable to assume that $hypothesis? $options",`

`"3_1"  : "$premise Considering the information provided, can we draw the conclusion that $hypothesis? $options",`
`"3_5"  : "$premise \n Can we deduce the following? $hypothesis $options",`
`"3_8"  : "Can we infer the following hypothesis based on the given $premise? Hypothesis: $hypothesis $options",`
`"3_10" : "$premise Assuming $hypothesis, what are the $options to consider?",`

`"5_1"  : "Can we infer whether $premise supports the hypothesis that $hypothesis? $options ",`
`"5_3"  : "$premise Can we deduce the following outcome? $hypothesis $options", `
`"5_8"  : "Can we draw a plausible hypothesis based on the given context? Context: $premise Hypothesis: $hypothesis $options",`
`"5_10" : "$premise \n Question: Can we infer the following? $hypothesis $options ",`

`"8_1"  : "Can we draw a plausible hypothesis from the given context? Context: $premise  Hypothesis: $hypothesis $options",`
`"8_3"  : "Can we infer the following outcome from the given scenario? Scenario: $premise \n Outcome: $hypothesis \n $options",`
`"8_5"  : "Can we derive the following hypothesis based on the given $premise? Furthermore, can we deduce the validity of $hypothesis considering $options?",`
`"8_10"  : "Considering the context outlined in $premise, can we deduce $hypothesis? Evaluate the following options: $options",`

`"10_1" : "Given the information provided, is it reasonable to infer that $hypothesis based on the $premise? $options",`
`"10_3" : "$premise \n Can we infer that $hypothesis? $options", `
`"10_5" : "$premise \n Can we deduce the following? $hypothesis $options",`
`"10_8" : "Given the context: $premise Is it reasonable to conclude: $hypothesis?$options"`

    "metrics": {
        "f1": 0.6470588235294118,
        "precision": 0.6346153846153846,
        "recall": 0.66
    }

| **Metrics**    | F1-score | Precision | Recall | Notes |
|:-------------- |:--:|:--:|:--:|:--:|
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_1-3   | 0.66 | 0.54 | 0.84 | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_1-5   | - | - | - | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_1-8   | 0.64 | 0.54 | 0.79 | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_1-10  | 0.67 | 0.56 | 0.82 | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_3-1   | 0.65 | 0.63 | 0.66 | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_3-5   | - | - | - | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_3-8   | - | - | - | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_3-10  | - | - | - | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_5-1   | - | - | - | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_5-3   | - | - | - | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_5-8   | - | - | - | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_5-10  | - | - | - | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_8-1   | - | - | - | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_8-3   | - | - | - | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_8-5   | - | - | - | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_8-10  | - | - | - | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_10-1  | - | - | - | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_10-3  | - | - | - | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_10-5  | - | - | - | - |
| qCammel-70-x-GPTQ_(apl{t5})_(gen)_10-8  | - | - | - | - |
---

#### Other notes

Max token limit per model:

##### NVIDIA A100-PCIE-40GB 
[qCammel-13-GPTQ] - 4096 tokens
[qCammel-70-x-GPTQ] - 2600 tokens (Can use more tokens than 4-bit because doesn't use group size)
[qCammel-70-x-GPTQ-4bit-32g] - 1250 tokens
[qCammel-70-x-GPTQ-4bit-64g] - 1650 tokens
[qCammel-70-x-GPTQ-4bit-128g] - 2050 tokens
[qCammel-70-x-GPTQ-3bit-1g] - 4096 tokens
[qCammel-70-x-GPTQ-3bit-128g] - 4096 tokens