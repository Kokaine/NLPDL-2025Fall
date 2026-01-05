import json
import os
import numpy as np
import matplotlib.pyplot as plt
from vllm import LLM, SamplingParams
from data.drgrpo_grader import r1_zero_reward_fn


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_gsm8k_answer(answer: str) -> str:
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip()

def calculate_avg_logprob(output_obj):
    """
    Calculates the normalized log-likelihood (average log-prob per token).
    vLLM returns logprobs as a list of dicts (one per token position).
    """
    if not output_obj.logprobs:
        return -float('inf')
    
    total_logprob = 0.0
    for i, token_logprob_dict in enumerate(output_obj.logprobs):
        token_id = output_obj.token_ids[i]
        val = token_logprob_dict[token_id]
        if hasattr(val, 'logprob'):
            total_logprob += val.logprob
        else:
            total_logprob += val

    return total_logprob / len(output_obj.token_ids)

def main():
    MODEL_PATH = "Qwen/Qwen3-0.6B"
    TEST_DATA_PATH = "data/gsm8k/test.jsonl"
    OUTPUT_FILE = "best_of_n_results.json"
    PROMPT_FILE = "data/r1_zero.prompt"
    MAX_N = 16
    N_VALUES = [1, 4, 8, 16]
    
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        PROMPT_TEMPLATE = f.read()

    print(f"Loading data from {TEST_DATA_PATH}...")
    examples = load_jsonl(TEST_DATA_PATH)

    prompts = [PROMPT_TEMPLATE.format(question=ex['question']) for ex in examples]

    print(f"Initializing LLM with N={MAX_N}...")
    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=0.95,
        max_model_len=2048,
        tensor_parallel_size=1
    )
    
    sampling_params = SamplingParams(
        n=MAX_N,
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        logprobs=1 
    )

    print("Generating responses...")
    request_outputs = llm.generate(prompts, sampling_params)
    metrics = {n: {'pass': 0, 'best': 0} for n in N_VALUES}
    total_examples = len(examples)

    print("Evaluating metrics...")
    for i, req_output in enumerate(request_outputs):
        ground_truth = extract_gsm8k_answer(examples[i]['answer'])
        candidates = req_output.outputs
        candidate_data = []

        for cand in candidates:
            reward = r1_zero_reward_fn(cand.text, ground_truth)
            format_reward = reward["format_reward"]
            answer_reward = reward["answer_reward"]
            is_correct = (answer_reward == 1.0)
            score = calculate_avg_logprob(cand)
            
            candidate_data.append({
                'is_correct': is_correct,
                'score': score,
                'text': cand.text
            })

        for n in N_VALUES:
            subset = candidate_data[:n]
            
            if any(c['is_correct'] for c in subset):
                metrics[n]['pass'] += 1
            
            best_candidate = max(subset, key=lambda x: x['score'])
            if best_candidate['is_correct']:
                metrics[n]['best'] += 1

    print("\n--- Results ---")
    final_stats = {}
    for n in N_VALUES:
        pass_acc = metrics[n]['pass'] / total_examples
        best_acc = metrics[n]['best'] / total_examples
        final_stats[n] = {"pass_at_n": pass_acc, "best_at_n": best_acc}
        print(f"N={n}: pass@N = {pass_acc:.4f}, Best@N = {best_acc:.4f}")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_stats, f, indent=2)

    ns = N_VALUES
    pass_scores = [final_stats[n]['pass_at_n'] for n in ns]
    best_scores = [final_stats[n]['best_at_n'] for n in ns]

    plt.figure(figsize=(8, 5))
    plt.plot(ns, pass_scores, marker='o', label='pass@N (Oracle Selection)')
    plt.plot(ns, best_scores, marker='x', label='Best@N (Log-prob Selection)')
    plt.xlabel('N (Number of Samples)')
    plt.ylabel('Accuracy')
    plt.title('pass@N vs Best@N Scaling')
    plt.legend()
    plt.grid(True)
    plt.savefig('scaling_plot.png')
    print("\nPlot saved to scaling_plot.png")

if __name__ == "__main__":
    main()