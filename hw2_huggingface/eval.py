# hw2_huggingface/eval.py

import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score_func
## TODO: import necessary libraries for evaluation metrics
device = "cuda" if torch.cuda.is_available() else "cpu"

def inference(model, tokenizer, inputs, max_length=512):
    '''Generate predictions from the model given inputs.'''
    model.eval()
    predictions = []

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        ## TODO: generate predictions using model and tokenizer
        for text in tqdm(inputs, desc="Generating"):
            model_inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length
            ).to(device)
        
            generated_ids = model.generate(
                **model_inputs, 
                max_new_tokens=512, 
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )
            input_len = model_inputs.input_ids.shape[1]
            generated_response = generated_ids[0][input_len:]
            decoded_text = tokenizer.decode(generated_response, skip_special_tokens=True)
            predictions.append(decoded_text)

    return predictions


def compute_bleu(references, candidates):
    ## TODO: Compute average BLEU score for reference and candidate pairs.
    bleu_result = sacrebleu.corpus_bleu(candidates, [references], tokenize="13a")
    return bleu_result.score / 100


def compute_rouge(references, candidates):
    ## TODO: Compute average ROUGE-L score for reference and candidate pairs.
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    f1_scores = []
    for ref, cand in zip(references, candidates):
        score = scorer.score(ref, cand)
        f1_scores.append(score['rougeL'].fmeasure)
        
    return np.mean(f1_scores)


def compute_bertscore(references, candidates, model="facebook/bart-large"):
    ## TODO: Compute average BERTScore for reference and candidate pairs.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    P, R, F1 = bert_score_func(
        candidates, 
        references, 
        model_type=model, 
        lang="en", 
        verbose=True,
        device=device,
        batch_size=16
    )
    
    return F1.mean().item()

def evaluate_model(model_name, dataset_name, split="test"):
    '''Evaluate the model on the given dataset and split.'''

    ## TODO: Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=dtype, 
        device_map="auto", # Automatically puts model on GPU
        trust_remote_code=True
    )

    ## TODO: Load dataset
    dataset = load_dataset(dataset_name, split=split)
    if "dialogue" in dataset.column_names:
        input_column = "dialogue"
        ref_column = "summary"
    elif "text" in dataset.column_names:
        input_column = "text"
        ref_column = "label" if "label" in dataset.column_names else "summary"
    else:
        raise ValueError(f"Could not identify input/output columns. Found: {dataset.column_names}")

    inputs = dataset[input_column]
    references = dataset[ref_column]

    candidates = inference(model, tokenizer, inputs)

    # Compute evaluation metrics
    bleu = compute_bleu(references, candidates)
    rouge_l = compute_rouge(references, candidates)
    bertscore = compute_bertscore(references, candidates)

    return {"BLEU": bleu, "ROUGE-L": rouge_l, "BERTScore": bertscore}


if __name__ == "__main__":
    model_name = "qwen-0.5b-samsum-merged"
    dataset_name = "knkarthick/samsum"

    results = evaluate_model(model_name, dataset_name)
    print("Evaluation Results:", results)
