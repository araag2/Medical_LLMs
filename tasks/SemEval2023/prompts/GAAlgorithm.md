# GA Algorithm

**Input:** 5 prompts to begin the procedure, composed of 4 different parts that need to be combined with the clinical trial information
input: data from the semeval task, composed of training and testing splits

### 1 - Evaluate the initial 5 prompts in the training data from the semeval task

### 2 - Repeat the following steps corresponding to an evolutionary procedure:

2.1 - Take the 5 best prompts from the previous iteration (or the 5 initial ones)

2.2 - For each of the 5 prompts, generate 1 new mutated version, thus resulting in a set of 10 prompts (initial plus new). an LLM is used to generate the mutations, using a fixed prompt to process individually each of the 4 parts that compose the prompt.

2.3 - With the set of 10 prompts from the previous step, generate 15 new prompts through a combination of two of the prompts sampled randomly from the set. this results in a set of 20 prompts (the 5 from step 2.2, plus the new 15). an LLM is used to generate the combinations, using a fixed prompt to process individually each of the 4 parts in each instance from the pair of prompts being combined.

2.4 - Evaluate the 20 prompts in the set resulting from the previous step, using the training data from the semeval task. take the 5 best prompts from the set.

2.5 - If one of the 5 new best prompts has a better evaluation score than the best prompt from the initial set of 5, then continue the procedure again from step 2.

2.6 - Considering a patience of one, built a new dataset of 20 prompts considering the five initial prompts, and repeating the procedures from steps 2.2 and 2.3.

2.7 - Evaluate the new set of 20 prompts resulting from the previous step, using the training data from the semeval task. take the 5 best prompts from the set.

2.8 - If one of the 5 new best prompts has a better evaluation score than the best prompt from the initial set of 5, then continue the procedure again from step 2. otherwise, stop the iteration and keep the best result/prompt.

3 - With basis on the best prompt resulting from the evolutionary procedure, evaluate the results on the testing split from the SemEval task. report this best result, together with the number of iterations that were required, and together with the evaluation score also on the test split.

4 - Taking the best prompt resulting from the previous procedure, fine-tune an LLM to the semeval task, using the training data.

5 - Evaluate the performance of the fine-tuned model on the testing split from the semeval task, comparing with the results obtained for step 3.

#### Prompt to Mutate Instructions

Consider the problem of re-writing a textual instruction, in which the objective is to rephrase the description while keeping the exact same meaning. The re-written instruction can either be shorter, summarizing the main points while keeping consistency with the original intent, or it can be made longer, by adding definitions and further clarifications while at the same time avoiding the inclusion of incorrect information. The re-written instruction should be concise and direct, and it should inform the execution of the task in a clearer way than the original instruction. Considering the aforementioned task description, re-write the textual instruction show next in quotes.

#### Prompt to Combine Instructions

Consider the problem of combining two different textual instructions, pertaining to the same task. The objective is to rephrase the main information common to the two descriptions, while keeping their meaning and intent. The combined instruction can either be shorter, summarizing the main points while keeping consistency with the original intent, or it can be made longer, by adding definitions and further clarifications while at the same time avoiding the inclusion of incorrect information. The combined instruction should be concise and direct, and it should inform the execution of the task in a clearer way than the original instructions. Considering the aforementioned task description, combine the two textual instructions show next in quotes.

#### 5 Prompts describing the task at hand

Consider the task of determining semantic entailment relations between individual sections of Clinical Trial Reports (CTRs) and statements made by clinical domain experts. Note that CTRs outline the methodology and findings of a clinical trial, which are conducted to assess the effectiveness and safety of new treatments. Each trial involves 1-2 patient groups, called cohorts or arms, and these groups may receive different treatments, or have different baseline characteristics. The complete CTRs contain 4 sections, corresponding to (1) a list of the ELIGIBILITY CRITERIA corresponding to the conditions for patients to be allowed to take part in the clinical trial, (2) a description for the INTERVENTION that specifies the type, dosage, frequency, and duration of treatments being studied, (3) a summary of the RESULTS, detailing aspects such as the number of participants in the trial, the outcome measures, the units, and the conclusions, and (4) a list of ADVERSE EVENTS corresponding to signs and symptoms observed in patients during the clinical trial. In turn, the statements are sentences that make some type of claim about the information contained in one of the aforementioned sections, either considering a single CTR or comparing two CTRs. In order for the entailment relationship to be established, the claim in the statement should be related to the clinical trial information, it should be supported by the CTR, and it must not contradict the provided descriptions.

You are tasked with determining support relationships between individual sections of Clinical Trial Reports (CTRs) and clinical statements. CTRs detail the methodology and findings of clinical trials, assessing effectiveness and safety of new treatments. CTRs consist of 4 sections: (1) ELIGIBILITY CRITERIA listing conditions for patient participation, (2) INTERVENTION description specifying type, dosage, frequency, and duration of treatments, (3) RESULTS summary detailing participants, outcome measures, units, and conclusions, and (4) ADVERSE EVENTS listing signs and symptoms observed. Statements make claims about information in these sections, either for a single CTR or comparing two.

Evaluate the semantic entailment between individual sections of Clinical Trial Reports (CTRs) and statements issued by clinical domain experts. CTRs expound on the methodology and outcomes of clinical trials, appraising the efficacy and safety of new treatments. The statements, on the other hand, assert claims about the information within specific sections of CTRs, for a single CTR or comparative analysis of two. For entailment validation, the statement's claim should align with clinical trial information, find support in the CTR, and refrain from contradicting provided descriptions.

The objective is to examine semantic entailment relationships between individual sections of Clinical Trial Reports (CTRs) and statements articulated by clinical domain experts. CTRs elaborate on the procedures and findings of clinical trials, scrutinizing the effectiveness and safety of novel treatments. Each trial involves cohorts or arms exposed to distinct treatments or exhibiting diverse baseline characteristics. Comprehensive CTRs comprise four sections: (1) ELIGIBILITY CRITERIA delineating conditions for patient inclusion, (2) INTERVENTION particulars specifying type, dosage, frequency, and duration of treatments, (3) RESULTS summary encompassing participant statistics, outcome measures, units, and conclusions, and (4) ADVERSE EVENTS cataloging signs and symptoms observed. Statements posit claims regarding the information within these sections, either for a single CTR or in comparative analysis of two. To establish entailment, the statement's assertion should harmonize with clinical trial data, find substantiation in the CTR, and avoid contradiction with the provided descriptions.

Consider the problem of assessing semantic entailment connections between distinct sections of Clinical Trial Reports (CTRs) and statements put forth by clinical domain experts. To establish entailment, the statement's assertion should be supported from the CTR, not contradicting the provided descriptions. In brief, CTRs elucidate the procedures and findings of clinical trials, evaluating the efficacy and safety of emerging treatments. Complete CTRs encompass four sections: (1) ELIGIBILITY CRITERIA specifying conditions for patient inclusion, (2) INTERVENTION details on the type, dosage, frequency, and duration of treatments, (3) RESULTS summarizing the participant statistics, outcome measures, units, and conclusions, and (4) ADVERSE EVENTS listing observed signs and symptoms. Statements advance claims about the information within these sections, either for a single CTR or in a comparative analysis of two CTRs.

#### FIVE PROMPTS GIVING A DESCRIPTION OF CTR INFORMATION

The following descriptions correspond to the information in one of the Clinical Trial Report (CTR) sections.

The provided descriptions coincide with the content in a specific section of Clinical Trial Reports (CTRs), detailing relevant information to the trial.

The provided descriptions correspond to the content found in one of the four standard clinical trial report sections.

The provided descriptions pertain to the contents found within one of the sections of Clinical Trial Reports (CTRs).

The descriptions that follow correspond to the information contained in one of the standard sections of the clinical trial reports.

#### FIVE PROMPTS GIVING A DESCRIPTION OF STATEMENT INFORMATION

Consider also the following statement generated by a clinical domain expert, a clinical trial organizer, or a medical researcher.

Contemplate the ensuing statement formulated by a clinical expert or researcher.

Review the subsequent statement provided by an expert in clinical trials, attending to the medical terminology and carefully addressing any ambiguities.

Deliberate upon the subsequent statement formulated by an healthcare practitioner, a coordinator of clinical trials, or a medical researcher.

Reflect upon the ensuing statement crafted by an expert in clinical trials.

#### FIVE PROMPTS GIVING A DESCRIPTION FOR HOW THE ANSWER SHOULD BE PROVIDED

Answer YES or NO to the question of whether one can conclude the validity of the statement with basis on the clinical trial report information.

Indicate with either YES or NO whether it is possible to determine the validity of the statement based on the Clinical Trial Report (CTR) descriptions. An answer of YES means that the statement is supported by the CTR descriptions, not contradicting the provided information.

Provide a YES or NO response indicating if it's possible to assess the statement's validity based on the information presented in the clinical trial report descriptions. Do this by interpreting the medical terminology and the context in both the report and the statement, carefully addressing any ambiguities or gaps in the provided information.

Respond with either YES or NO to indicate whether it is possible to determine the statement's validity based on the Clinical Trial Report (CTR) information, with the statement being supported by the CTR data and not contradicting the provided descriptions.

Indicate with a YES or NO response whether it is possible to assess the statement's validity based on the clinical trial report data.

#### PROMPT FOR A COMPLETE EXAMPLE, ILLUSTRATING HOW THE 4 DIFFERENT PARTS SHOULD BE COMBINED WITH THE INFORMATION FROM A DATA INSTANCE

Consider the task of determining semantic entailment relations between individual sections of Clinical Trial Reports (CTRs) and statements made by clinical domain experts. Note that CTRs outline the methodology and findings of a clinical trial, which are conducted to assess the effectiveness and safety of new treatments. Each trial involves 1-2 patient groups, called cohorts or arms, and these groups may receive different treatments, or have different baseline characteristics. The complete CTRs contain 4 sections, corresponding to (1) a list of the ELIGIBILITY CRITERIA corresponding to the conditions for patients to be allowed to take part in the clinical trial, (2) a description for the INTERVENTION that specifies the type, dosage, frequency, and duration of treatments being studied, (3) a summary of the RESULTS, detailing aspects such as the number of participants in the trial, the outcome measures, the units, and the conclusions, and (4) a list of ADVERSE EVENTS corresponding to signs and symptoms observed in patients during the clinical trial. In turn, the statements are sentences that make some type of claim about the information contained in one of the aforementioned sections, either considering a single CTR or comparing two CTRs. In order for the entailment relationship to be established, the claim in the statement should be related to the clinical trial information, it should be supported by the CTR, and it must not contradict the provided descriptions.

The following descriptions correspond to the information in one of the sections of the Clinical Trial Reports (CTRs).

Primary Trial
INTERVENTION 1:
•  Letrozole, Breast Enhancement, Safety
•  Single arm of healthy postmenopausal women to have two breast MRI (baseline and post-treatment). Letrozole of 12.5 mg/day is given for three successive days just prior to the second MRI.

Secondary Trial
INTERVENTION 1: 
•  Healthy Volunteers
•  Healthy women will be screened for Magnetic Resonance Imaging (MRI) contraindications, and then undergo contrast injection, and SWIFT acquisition.
•  Magnetic resonance imaging: Patients and healthy volunteers will be first screened for MRI contraindications. The SWIFT MRI workflow will be performed as follows:

Consider also the following statement generated by a clinical domain expert, a clinical trial organizer, or a research oncologist.

The primary trial and the secondary trial both used MRI for their interventions.

Answer YES or NO to the question of whether one can conclude the validity of the statement with basis on the Clinical Trial Report (CTR) information.

