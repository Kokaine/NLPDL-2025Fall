import json
import os
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

def main():
    MODEL_PATH = "Qwen/Qwen3-0.6B"
    TEST_DATA_PATH = "data/gsm8k/test.jsonl"
    OUTPUT_FILE = "zero_shot_results.jsonl"
    PROMPT_FILE = "data/r1_zero.prompt"
    
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        PROMPT_TEMPLATE = f.read()

    print(f"Loading data from {TEST_DATA_PATH}...")
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: {TEST_DATA_PATH} not found.")
        return
        
    examples = load_jsonl(TEST_DATA_PATH)
    prompts = [PROMPT_TEMPLATE.format(question=ex['question']) for ex in examples]
    llm = LLM(model=MODEL_PATH)
    
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    

    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    stats = {"total": 0, "correct": 0, "format_error": 0, "wrong_answer": 0}

    print("Evaluating responses...")
    for i, output in enumerate(outputs):
        prompt_text = output.prompt
        generated_text = output.outputs[0].text
        ground_truth = extract_gsm8k_answer(examples[i]['answer'])

        reward = r1_zero_reward_fn(generated_text, ground_truth)

        if i < 5:
            print(f"\n--- DEBUG EXAMPLE {i} ---")
            print(f"GT Answer: [{ground_truth}]")
            print(f"Model Gen: {generated_text[-100:]}")
            print(f"Format Reward: {reward['format_reward']}")
            print(f"Answer Reward: {reward['answer_reward']}")

        format_reward = reward["format_reward"]
        answer_reward = reward["answer_reward"]
        
        result_entry = {
            "question": examples[i]['question'],
            "ground_truth": ground_truth,
            "generated_text": generated_text,
            "format_reward": format_reward,
            "answer_reward": answer_reward
        }
        results.append(result_entry)
        
        stats["total"] += 1
        if format_reward == 1 and answer_reward == 1:
            stats["correct"] += 1
        elif format_reward == 0:
            stats["format_error"] += 1
        elif format_reward == 1 and answer_reward == 0:
            stats["wrong_answer"] += 1

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