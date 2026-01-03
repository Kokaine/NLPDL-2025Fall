import json
import os
from vllm import LLM, SamplingParams
# Assuming the course environment has this module available
# If running locally without the specific course structure, you would need to mock this function.
try:
    from data.drgrpo_grader import r1_zero_reward_fn
except ImportError:
    print("Warning: Could not import r1_zero_reward_fn. Define a mock or ensure PYTHONPATH is correct.")
    # Mock for demonstration purposes if module is missing
    def r1_zero_reward_fn(model_output, ground_truth):
        # Returns (format_reward, answer_reward)
        # This is a placeholder logic
        if "<answer>" in model_output and "</answer>" in model_output:
            return 1.0, 0.0 # Placeholder
        return 0.0, 0.0

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    # --- Configuration ---
    MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct" # Replace with actual path to Qwen3-0.6B if local
    TEST_DATA_PATH = "data/gsm8k/test.jsonl"
    OUTPUT_FILE = "zero_shot_results.jsonl"
    
    # r1_zero prompt template from instructions
    PROMPT_TEMPLATE = (
        "A conversation between User and Assistant. The User takes a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in the mind and them provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, "
        "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
        "User: {question}\n"
        "Assistant: <think>"
    )

    # --- 1. Load Data ---
    print(f"Loading data from {TEST_DATA_PATH}...")
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: {TEST_DATA_PATH} not found.")
        return
        
    examples = load_jsonl(TEST_DATA_PATH)
    
    # --- 2. Format Prompts ---
    prompts = [PROMPT_TEMPLATE.format(question=ex['question']) for ex in examples]
    
    # --- 3. Initialize vLLM ---
    print(f"Initializing LLM: {MODEL_PATH}...")
    llm = LLM(model=MODEL_PATH)
    
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    
    # --- 4. Generate Outputs ---
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)
    
    # --- 5. Evaluate and Serialize ---
    results = []
    stats = {"total": 0, "correct": 0, "format_error": 0, "wrong_answer": 0}

    print("Evaluating responses...")
    for i, output in enumerate(outputs):
        prompt_text = output.prompt
        generated_text = output.outputs[0].text
        ground_truth = examples[i]['answer'] # GSM8K usually has 'answer' field
        
        # Use the provided reward function
        # Note: The prompt implies the reward function returns a boolean or score.
        # Often these graders return (format_score, accuracy_score) or just accuracy.
        # Based on part (b), we need to distinguish format vs answer correctness.
        # Assuming r1_zero_reward_fn returns (format_reward, answer_reward) or similar structure.
        # If it only returns boolean, we must parse manually to detect format errors.
        
        # We assume the signature is: r1_zero_reward_fn(completion, gold_answer) -> (format_reward, answer_reward)
        # If the provided grader works differently, adjust accordingly.
        format_reward, answer_reward = r1_zero_reward_fn(generated_text, ground_truth)
        
        result_entry = {
            "question": examples[i]['question'],
            "ground_truth": ground_truth,
            "generated_text": generated_text,
            "format_reward": format_reward,
            "answer_reward": answer_reward
        }
        results.append(result_entry)
        
        # Update Stats
        stats["total"] += 1
        if format_reward == 1 and answer_reward == 1:
            stats["correct"] += 1
        elif format_reward == 0:
            stats["format_error"] += 1
        elif format_reward == 1 and answer_reward == 0:
            stats["wrong_answer"] += 1

    # Write results to disk
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print("\n--- Evaluation Complete ---")
    print(f"Total Examples: {stats['total']}")
    print(f"Correct (Format 1, Answer 1): {stats['correct']} ({(stats['correct']/stats['total'])*100:.2f}%)")
    print(f"Format Errors (Format 0): {stats['format_error']} ({(stats['format_error']/stats['total'])*100:.2f}%)")
    print(f"Wrong Answers (Format 1, Answer 0): {stats['wrong_answer']} ({(stats['wrong_answer']/stats['total'])*100:.2f}%)")

if __name__ == "__main__":
    main()