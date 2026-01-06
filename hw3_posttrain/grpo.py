import sys
import os
import json
import random
import torch
import typer
import swanlab
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Literal, Optional
from unittest.mock import patch

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from utils import (
    compute_group_normalized_rewards,
    tokenize_prompt_and_output,
    get_response_log_probs,
    grpo_microbatch_train_step,
    masked_mean,
    extract_gsm8k_answer
)
from data.drgrpo_grader import r1_zero_reward_fn


@dataclass
class GRPOConfig:
    model_name_or_path: str = 'Qwen/Qwen3-0.6B'
    train_data_path: str = './data/gsm8k/train.jsonl'
    val_data_path: str = './data/gsm8k/test.jsonl'
    prompt_path: str = './data/r1_zero.prompt'
    output_dir: str = './checkpoints/grpo'
    
    # Training Hyperparams
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 128
    rollout_batch_size: int = 256
    train_batch_size: int = 256
    epochs_per_rollout: int = 1
    group_size: int = 8
    loss_type: str = "reinforce_with_baseline"
    use_std_normalization: bool = True
    cliprange: float = 0.2
    advantage_eps: float = 1e-6
    
    # Generation
    temp: float = 1.0
    max_tokens: int = 1024
    min_tokens: int = 4
    
    # System
    seed: int = 42
    gpu_memory_utilization: float = 0.85
    train_device: str = 'cuda:0'
    eval_device: str = 'cuda:1'
    
    swanlab_project: str = 'NLPDL-grpo'
    swanlab_run_name: str = 'grpo-run'


def load_data(path: str):
    questions, answers = [], []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data['question'])
            answers.append(extract_gsm8k_answer(data['answer']))
    return questions, answers

def load_prompt_template(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()

def setup_vllm(model_path: str, device: str, seed: int, mem_util: float) -> LLM:
    """Initialize vLLM with memory profiling patches for single-GPU setups."""
    from vllm.model_executor import set_random_seed
    set_random_seed(seed)
    
    with patch("torch.distributed.get_world_size", return_value=1), \
         patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None):
        return LLM(
            model=model_path,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=mem_util,
        )

def sync_policy_to_vllm(policy: torch.nn.Module, vllm_instance: LLM):
    """Transfer weights from PyTorch training model to vLLM inference engine."""
    vllm_model = vllm_instance.llm_engine.model_executor.driver_worker.model_runner.model
    vllm_model.load_weights(policy.state_dict().items())


def perform_rollout(
    vllm_model: LLM, 
    questions: List[str], 
    answers: List[str], 
    template: str, 
    cfg: GRPOConfig
):
    """Generates responses using vLLM."""
    # Sample batch
    indices = random.sample(range(len(questions)), cfg.rollout_batch_size // cfg.group_size)
    batch_qs = [questions[i] for i in indices]
    batch_gts = [answers[i] for i in indices]
    prompts = [template.replace('{question}', q) for q in batch_qs]

    # Generate
    sampling_params = SamplingParams(
        n=cfg.group_size, temperature=cfg.temp, top_p=1.0, 
        max_tokens=cfg.max_tokens, min_tokens=cfg.min_tokens,
        stop=["</answer>"], include_stop_str_in_output=True
    )
    outputs = vllm_model.generate(prompts, sampling_params)
    # Flatten results
    flat_responses, flat_prompts, flat_gts = [], [], []
    for output, gt, prompt in zip(outputs, batch_gts, prompts):
        for gen in output.outputs:
            flat_responses.append(gen.text)
            flat_gts.append(gt)
            flat_prompts.append(prompt)
            
    return flat_responses, flat_prompts, flat_gts

def compute_metrics_and_rewards(responses, prompts, gts, cfg: GRPOConfig):
    """Calculates rewards and advantages."""
    advantages, raw_rewards, metadata = compute_group_normalized_rewards(
        reward_fn=r1_zero_reward_fn,
        rollout_responses=responses,
        repeated_ground_truths=gts,
        group_size=cfg.group_size,
        advantage_eps=cfg.advantage_eps,
        normalize_by_std=cfg.use_std_normalization
    )
    return advantages.unsqueeze(1), raw_rewards.unsqueeze(1), metadata

def train_optimization_step(
    policy, optimizer, tokenizer, 
    input_ids, labels, response_mask, advantages, raw_rewards, old_log_probs,
    cfg: GRPOConfig
):
    """Runs the inner training loop (epochs and micro-batches)."""
    policy.train()
    micro_bs = cfg.train_batch_size // cfg.gradient_accumulation_steps
    n_batches = cfg.rollout_batch_size // micro_bs

    for _ in range(cfg.epochs_per_rollout):
        perm = torch.randperm(cfg.rollout_batch_size)
        
        for i in range(n_batches):
            idx = perm[i*micro_bs : (i+1)*micro_bs]
            
            mb_inputs = input_ids[idx].to(cfg.train_device)
            mb_labels = labels[idx].to(cfg.train_device)
            mb_mask = response_mask[idx].to(cfg.train_device)
            mb_adv = advantages[idx].to(cfg.train_device)
            mb_raw = raw_rewards[idx].to(cfg.train_device)
            mb_old_lp = old_log_probs[idx].to(cfg.train_device) if old_log_probs is not None else None

            # Forward
            log_prob_data = get_response_log_probs(
                model=policy, input_ids=mb_inputs, labels=mb_labels, return_token_entropy=True
            )
            
            loss, _ = grpo_microbatch_train_step(
                policy_log_probs=log_prob_data['log_probs'],
                response_mask=mb_mask,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                loss_type=cfg.loss_type,
                raw_rewards=mb_raw if cfg.loss_type == "no_baseline" else None,
                advantages=mb_adv if cfg.loss_type != "no_baseline" else None,
                old_log_probs=mb_old_lp,
                cliprange=cfg.cliprange
            )

            if (i + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                avg_ent = masked_mean(log_prob_data['token_entropy'], mb_mask, dim=None).item()
                
                swanlab.log({
                    "train/loss": loss.item(), 
                    "train/entropy": avg_ent
                })

def evaluate(vllm_model, questions, answers, template, cfg: GRPOConfig, step: int):
    """Runs evaluation on validation set."""
    indices = random.sample(range(len(questions)), min(512, len(questions)))
    prompts = [template.replace('{question}', questions[i]) for i in indices]
    gts = [answers[i] for i in indices]

    params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True)
    outputs = vllm_model.generate(prompts, params)
    
    total_acc, total_fmt, total_ans = 0, 0, 0
    for out, gt in zip(outputs, gts):
        r = r1_zero_reward_fn(out.outputs[0].text, gt)
        total_acc += r['reward']
        total_fmt += r.get('format_reward', 0)
        total_ans += r.get('answer_reward', 0)

    n = len(prompts)
    
    swanlab.log({
        "eval/accuracy": total_acc/n,
        "eval/format": total_fmt/n,
        "eval/answer": total_ans/n,
        "eval_step": step
    })
    print(f"Eval Step {step}: Acc={total_acc/n:.2f}")

app = typer.Typer()

@app.command()
def main(
    config_path: Optional[str] = None,
    learning_rate: float = 1e-5,
    n_grpo_steps: int = 200,
):
    cfg = GRPOConfig(learning_rate=learning_rate, n_grpo_steps=n_grpo_steps)
    
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    swanlab.init(
        project=cfg.swanlab_project,
        experiment_name=cfg.swanlab_run_name,
        config=cfg.__dict__
    )


    train_q, train_a = load_data(cfg.train_data_path)
    val_q, val_a = load_data(cfg.val_data_path)
    prompt_tpl = load_prompt_template(cfg.prompt_path)

    print("Initializing Models...")
    policy = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(cfg.train_device)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    vllm_model = setup_vllm(cfg.model_name_or_path, cfg.eval_device, cfg.seed, cfg.gpu_memory_utilization)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95))

    # GRPO Loop
    print(f"Starting GRPO for {cfg.n_grpo_steps} steps...")
    for step in range(cfg.n_grpo_steps):
        
        # 1. Rollout
        sync_policy_to_vllm(policy, vllm_model)
        responses, prompts, gts = perform_rollout(vllm_model, train_q, train_a, prompt_tpl, cfg)
        
        # 2. Rewards
        advantages, raw_rewards, meta = compute_metrics_and_rewards(responses, prompts, gts, cfg)
        
        swanlab.log({"train/mean_reward": meta['group_means'].mean().item(), "train_step": step})
        
        # 3. Prepare Data (Tokenize)
        tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
        input_ids, labels, mask = tokenized['input_ids'], tokenized['labels'], tokenized['response_mask']

        # 4. Off-policy prep
        old_log_probs = None
        if cfg.loss_type == "grpo_clip":
            with torch.inference_mode():
                out = get_response_log_probs(policy, input_ids.to(cfg.train_device), labels.to(cfg.train_device), False)
                old_log_probs = out['log_probs'].detach()

        # 5. Optimization
        train_optimization_step(
            policy, optimizer, tokenizer, 
            input_ids, labels, mask, advantages, raw_rewards, old_log_probs, 
            cfg
        )

        # 6. Evaluation & Save
        if (step + 1) % 5 == 0:
            evaluate(vllm_model, val_q, val_a, prompt_tpl, cfg, step)
        
        if (step + 1) % 10 == 0:
            policy.save_pretrained(f"{cfg.output_dir}/step_{step+1}")
            tokenizer.save_pretrained(f"{cfg.output_dir}/step_{step+1}")

    swanlab.finish()

if __name__ == "__main__":
    app()