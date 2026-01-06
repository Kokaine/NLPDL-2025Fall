from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
import torch.nn.functional as F

def extract_gsm8k_answer(answer: str) -> str:
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip()

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    assert len(prompt_strs) == len(output_strs), 'invalid input or label dimensions!'
    input_prompts_ids, output_ids = [], []
    for p in prompt_strs:
        p_id = tokenizer.encode(p, add_special_tokens=False)
        input_prompts_ids.append(torch.tensor(p_id))
    for o in output_strs:
        o_id = tokenizer.encode(o, add_special_tokens=False)
        output_ids.append(torch.tensor(o_id))
    prompt_and_output_lens = [len(promp)+len(out) for promp, out in zip(input_prompts_ids, output_ids)]
    D_output = max(prompt_and_output_lens) - 1
    #padding
    paded_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
    input_ids = []
    labels = []
    response_mask = []
    for p_id, o_id in zip(input_prompts_ids, output_ids):
        input_id = torch.cat((p_id, o_id, torch.tensor([tokenizer.eos_token_id])), dim=-1)
        response_m = torch.cat((torch.zeros_like(p_id).to(dtype=torch.bool), torch.ones_like(o_id).to(dtype=torch.bool), torch.tensor([False])), dim=-1)
        slice_input_id = input_id[:-1]
        slice_output_id = input_id[1:]
        slice_response_m = response_m[1:]
        pad_len = D_output - slice_input_id.shape[0]

        padded_input_id = F.pad(slice_input_id, (0, pad_len), value=paded_val)
        padded_output_id = F.pad(slice_output_id, (0, pad_len), value=paded_val)
        response_mask_padded = F.pad(slice_response_m, (0, pad_len), value=False)
        
        input_ids.append(padded_input_id)
        labels.append(padded_output_id)
        response_mask.append(response_mask_padded)
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'response_mask': torch.stack(response_mask)
    }

def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """

    raw_rewards_list = []
    for response, truth in zip(rollout_responses, repeated_ground_truths):
        raw_reward = reward_fn(response, truth)
        reward = raw_reward.get("reward", sum(raw_reward.values()))
        raw_rewards_list.append(float(reward))

    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    if raw_rewards.size(0) % group_size != 0:
        raise ValueError(f"Total responses ({raw_rewards.size(0)}) not divisible by group size ({group_size})")
        
    num_prompts = raw_rewards.size(0) // group_size
    rewards_matrix = raw_rewards.view(num_prompts, group_size)
    group_means = rewards_matrix.mean(dim=1, keepdim=True)
    group_stds = rewards_matrix.std(dim=1, keepdim=True)


    if normalize_by_std:
        advantages_matrix = (rewards_matrix - group_means) / (group_stds + advantage_eps)
    else:
        advantages_matrix = rewards_matrix - group_means

    advantages = advantages_matrix.flatten()
    
    metadata = {
        "group_means": group_means.flatten(),
        "group_stds": group_stds.flatten()
    }

    return advantages, raw_rewards, metadata

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    normed_logits = F.softmax(logits, dim=-1)
    log_p = torch.log(normed_logits)
    return -torch.sum(normed_logits*log_p, dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    pred_logits = model(input_ids).logits
    log_probs_all = F.log_softmax(pred_logits, dim=-1)

    labels_expanded = labels.unsqueeze(-1)  # (batch_size, seq_length, 1)
    log_probs = torch.gather(log_probs_all, dim=-1, index=labels_expanded).squeeze(-1)

    if return_token_entropy:
        entropy = compute_entropy(pred_logits)
    else:
        entropy = None

    return {
        'log_probs': log_probs,
        'token_entropy': entropy
    }

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    naive_loss = -raw_rewards_or_advantages * policy_log_probs
    
    return naive_loss


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """

    ratio = torch.exp(policy_log_probs - old_log_probs)
    unclipped = ratio * advantages
    

    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    clipped = clipped_ratio * advantages
    
    loss = -torch.min(unclipped, clipped)

    with torch.no_grad():
        clip_mask = (clipped < unclipped).float()
        clip_fraction = clip_mask.mean()
    
    metadata = {
        "clip_fraction": clip_fraction
    }
    
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """

    metadata: Dict[str, torch.Tensor] = {}
    loss: torch.Tensor

    if loss_type == "no_baseline": 
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs
        )

    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs
        )

    elif loss_type == "grpo_clip":
        loss, grpo_metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange
        )
        
        metadata.update(grpo_metadata)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss, metadata


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """

    mask_float = mask.float()
    masked_sum = (tensor * mask_float).sum(dim=dim)
    mask_count = mask_float.sum(dim=dim)
    
    return masked_sum / mask_count

    
def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """

    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange
    )
    
    per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)

    loss = per_example_loss.mean()
    
    scaled_loss = loss / gradient_accumulation_steps
    
    scaled_loss.backward()
    
    return loss.detach(), metadata