import os
import torch
import typer
import random
import numpy as np
from typing import Literal, List, Dict
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

# --- Import your implementations from previous problems ---
# Replace 'src.grpo' with the actual file name where you saved your functions
try:
    from utils import (
        compute_group_normalized_rewards,
        grpo_microbatch_train_step,
        masked_mean
    )
    from data.drgrpo_grader import r1_zero_reward_fn
except ImportError:
    print("WARNING: Could not import GRPO functions. Ensure your PYTHONPATH is correct.")
    # You might want to copy-paste your implementations here if imports fail

app = typer.Typer()

# --- Dataset Helper ---
class PromptDataset(Dataset):
    def __init__(self, data_path, prompt_template_path):
        import json
        self.data = []
        with open(data_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))
        
        with open(prompt_template_path, "r") as f:
            self.template = f.read()
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = self.template.format(question=item["question"])
        ground_truth = item["answer"]
        return prompt, ground_truth

# --- Main Training Loop ---
@app.command()
def train(
    # Hyperparameters from Source 16
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.85,
    loss_type: str = "reinforce_with_baseline", # Type: Literal["no_baseline", ...]
    use_std_normalization: bool = True,
    
    # System Paths
    model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
    train_data_path: str = "data/gsm8k/train.jsonl", # Or MATH
    test_data_path: str = "data/gsm8k/test.jsonl",
    prompt_file: str = "data/r1_zero.prompt",
    output_dir: str = "checkpoints/grpo_run"
):
    # [cite_start]1. Sanity Checks & Setup [cite: 16]
    assert train_batch_size % gradient_accumulation_steps == 0, "Train BS must be divisible by Grad Accum"
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    
    assert rollout_batch_size % group_size == 0, "Rollout BS must be divisible by Group Size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    
    assert train_batch_size >= group_size, "Train BS should generally be >= Group Size"
    
    # Create Output Directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Initialize Models
    print("Initializing vLLM for Rollouts...")
    # Note: If running on 2 GPUs, set tensor_parallel_size or CUDA_VISIBLE_DEVICES externally
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=4096 # Adjust based on memory
    )
    sampling_params = SamplingParams(
        n=group_size,
        temperature=sampling_temperature,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        [cite_start]stop=["</answer>", "</answer>\n"] # Stop at second answer tag [cite: 16]
    )
    
    print("Initializing Policy Model for Training...")
    # Load model in bfloat16 for stability
    policy = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # [cite_start]3. Optimizer [cite: 16]
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95)
    )

    # 4. Data Loaders
    train_dataset = PromptDataset(train_data_path, prompt_file)
    val_dataset = PromptDataset(test_data_path, prompt_file)
    
    # Simple infinite iterator for training prompts
    def infinite_prompt_loader(dataset, batch_size):
        while True:
            idxs = np.random.choice(len(dataset), batch_size, replace=False)
            yield [dataset[i] for i in idxs]

    prompt_iter = infinite_prompt_loader(train_dataset, n_prompts_per_rollout_batch)

    # --- Training Loop ---
    print(f"Starting GRPO Training for {n_grpo_steps} steps...")
    
    for step in range(1, n_grpo_steps + 1):
        policy.eval() # Ensure inference mode for rollouts
        
        # A. Rollout Phase (vLLM)
        prompts_batch = next(prompt_iter) # List of (prompt, ground_truth) tuples
        prompts = [p[0] for p in prompts_batch]
        ground_truths = [p[1] for p in prompts_batch]
        
        # Generate outputs
        # vLLM returns RequestOutput objects containing group_size completions
        request_outputs = llm.generate(prompts, sampling_params)
        
        # Flatten outputs for processing
        flat_prompts = []
        flat_responses = []
        flat_ground_truths = []
        
        for i, req_output in enumerate(request_outputs):
            gt = ground_truths[i]
            for completion in req_output.outputs:
                flat_prompts.append(req_output.prompt)
                flat_responses.append(completion.text)
                flat_ground_truths.append(gt)

        # B. Compute Rewards & Advantages
        # Using your implementation from Problem 3
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=flat_responses,
            repeated_ground_truths=flat_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization
        )
        
        # Move to GPU
        advantages = advantages.to(policy.device)
        raw_rewards = raw_rewards.to(policy.device)
        
        # [cite_start]Log Train Rewards [cite: 17]
        avg_reward = raw_rewards.mean().item()
        print(f"Step {step} | Train Reward: {avg_reward:.4f} | Adv Mean: {advantages.mean().item():.4f}")

        # C. Off-Policy Preparation
        # If using GRPO-Clip or multiple epochs, we need old_log_probs. 
        # Since we just generated, the "current" policy IS the "old" policy for the first epoch.
        old_log_probs = None
        if loss_type == "grpo_clip" or epochs_per_rollout_batch > 1:
            with torch.no_grad():
                # We must re-run forward pass with PyTorch model to get log_probs matching the graph
                # Tokenize (Prompt + Response)
                inputs = tokenizer(
                    [p + r for p, r in zip(flat_prompts, flat_responses)],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(policy.device)
                
                outputs = policy(**inputs)
                # Compute log_probs from logits
                # Gather log probs at specific token indices
                all_log_probs = outputs.logits.log_softmax(dim=-1)
                
                # Gather log_prob of the actual token taken
                # Shift logits: logits[i] predicts token[i+1]
                input_ids = inputs["input_ids"]
                target_ids = input_ids[:, 1:] 
                all_log_probs = all_log_probs[:, :-1, :]
                
                # Gather: shape (Batch, Seq)
                old_log_probs = torch.gather(all_log_probs, -1, target_ids.unsqueeze(-1)).squeeze(-1)
                
                # We also need a mask for the response part only
                # Identify where response starts (simple heuristic or finding prompt length)
                # For simplicity here: we assume padding is on the right or handle prompt masking carefully
                # A robust way is to mask out the prompt tokens.
                # (Implementation detail: create a mask 0 for prompt, 1 for response)
                
        # D. Optimization Loop
        policy.train()
        
        # Create full training data indices
        indices = torch.randperm(len(flat_responses))
        
        for epoch in range(epochs_per_rollout_batch):
            for i in range(0, len(indices), train_batch_size):
                batch_indices = indices[i : i + train_batch_size]
                
                # Further split into microbatches
                optimizer.zero_grad()
                
                batch_loss = 0.0
                batch_grad_norm = 0.0
                
                for j in range(0, len(batch_indices), micro_train_batch_size):
                    mb_indices = batch_indices[j : j + micro_train_batch_size]
                    
                    # Prepare inputs for this microbatch
                    mb_prompts = [flat_prompts[k] for k in mb_indices]
                    mb_responses = [flat_responses[k] for k in mb_indices]
                    
                    inputs = tokenizer(
                        [p + r for p, r in zip(mb_prompts, mb_responses)],
                        return_tensors="pt", 
                        padding=True,
                        truncation=True
                    ).to(policy.device)
                    
                    # Create Response Mask (1 for response, 0 for prompt/padding)
                    # We need to find where prompt ends. 
                    # Note: This is simplified. Robust code matches prompt_ids.
                    prompt_inputs = tokenizer(mb_prompts, return_tensors="pt", padding=True)
                    prompt_lens = prompt_inputs["attention_mask"].sum(dim=1)
                    
                    response_mask = torch.zeros_like(inputs["input_ids"][:, 1:], dtype=torch.float32) # Shifted for next-token
                    for idx, p_len in enumerate(prompt_lens):
                        # The prompt length might vary due to tokenization differences when concat
                        # This is a critical alignment step in real implementation.
                        # Approximate: set 1 after p_len-1
                        response_mask[idx, int(p_len)-1:] = 1.0
                        # Mask out padding
                        response_mask[idx] = response_mask[idx] * inputs["attention_mask"][idx, 1:]

                    # Forward Pass (New Policy)
                    outputs = policy(**inputs)
                    logits = outputs.logits[:, :-1, :]
                    input_ids = inputs["input_ids"][:, 1:]
                    
                    new_log_probs = torch.gather(logits.log_softmax(-1), -1, input_ids.unsqueeze(-1)).squeeze(-1)
                    
                    # Slice Data for Microbatch
                    mb_raw_rewards = raw_rewards[mb_indices].unsqueeze(1) if raw_rewards is not None else None
                    mb_advantages = advantages[mb_indices].unsqueeze(1) if advantages is not None else None
                    
                    # Handle Old Log Probs (Alignment is tricky with padding, assume `old_log_probs` matches `new_log_probs` shape if re-tokenized identically)
                    mb_old_log_probs = old_log_probs[mb_indices] if old_log_probs is not None else None
                    
                    # Compute Loss & Backward
                    loss, loss_meta = grpo_microbatch_train_step(
                        policy_log_probs=new_log_probs,
                        response_mask=response_mask,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        loss_type=loss_type,
                        raw_rewards=mb_raw_rewards,
                        advantages=mb_advantages,
                        old_log_probs=mb_old_log_probs,
                        [cite_start]cliprange=1.0 # [cite: 16]
                    )
                    
                    batch_loss += loss.item()
                
                # [cite_start]Clip Gradients [cite: 16]
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                
                # Log Step Metrics (Optional: Log to wandb/tensorboard here)
                # print(f"  Loss: {batch_loss:.4f} | Grad Norm: {grad_norm:.4f}")

        # E. Validation
        # [cite_start]Routinely log validation rewards (e.g., every 5-10 steps) [cite: 16]
        if step % 10 == 0:
            print(f"--- Validation Step {step} ---")
            run_validation(llm, val_dataset, r1_zero_reward_fn, limit=1024)
            # Save Checkpoint
            policy.save_pretrained(f"{output_dir}/step_{step}")
            tokenizer.save_pretrained(f"{output_dir}/step_{step}")

def run_validation(llm, dataset, reward_fn, limit=1024):
    [cite_start]"""Evaluate on >= 1024 validation examples [cite: 16]"""
    # Sample indices
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:limit]
    
    prompts = [dataset[i][0] for i in indices]
    gts = [dataset[i][1] for i in indices]
    
    # Validation Sampling (Greedy or low temp usually, but prompt implies similar settings)
    val_params = SamplingParams(temperature=0.0, max_tokens=1024, stop=["</answer>"])
    
    outputs = llm.generate(prompts, val_params)
    
    correct_count = 0
    for i, out in enumerate(outputs):
        response = out.outputs[0].text
        # Clean response if necessary
        clean_response = response.replace("<answer> ", "<answer>").replace(" </answer>", "</answer>")
        reward_dict = reward_fn(clean_response, gts[i])
        
        # Handle dict/tuple return
        if isinstance(reward_dict, dict):
            acc = reward_dict.get("answer_reward", 0)
        else:
            acc = reward_dict[1] # Assuming tuple (format, answer)
            
        if acc == 1.0:
            correct_count += 1
            
    acc_percentage = (correct_count / len(indices)) * 100
    print(f"Validation Accuracy: {acc_percentage:.2f}%")

if __name__ == "__main__":
    app()