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


### Baseline Results

#### Train Set (0-shot)

| **Metrics**    | F1-score | Precision | Recall | Notes |
|:-------------- |:--:|:--:|:--:|:--:|
| flanT5-small | 0.02 | 0.50 | 0.50 | Always Contradiction |
| flanT5-base | 0.32 | 0.50 | 0.23 | - |
| flanT5-large | 0.53 | 0.56 | 0.49 | - |
| flanT5-xl | 0.67 | 0.59 | 0.77 | - |
| flanT5-xxl | 0.69 | 0.61 | 0.79 | - |

#### Dev Set (0-shot)

| **Metrics**    | F1-score | Precision | Recall | Notes |
|:-------------- |:--:|:--:|:--:|:--:|
| flanT5-small | 0.00 | 0.00 | 0.00 | Always Contradiction |
| flanT5-base | 0.34 | 0.55 | 0.25 | - |
| flanT5-large | 0.57 | 0.61 | 0.53 | - |
| flanT5-xl | 0.69 | 0.61 | 0.79 | - |
| flanT5-xxl | 0.67 | - | - | - |

#### Dev Set (fine-tuned)

| **Metrics**    | F1-score | Precision | Recall | Notes |
|:-------------- |:--:|:--:|:--:|:--:|
| flanT5-xl | 0.754 | 0.59 | 0.831 | - |