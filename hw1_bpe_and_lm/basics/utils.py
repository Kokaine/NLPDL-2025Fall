from __future__ import annotations

import os
import math
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor

from my_train_bpe import train_bpe
from tokenizer import Tokenizer
from modules import Linear, Embedding, RMSNorm, SwiGLU, RoPE, softmax
from modules import scaled_dot_product_attention as sdpa
from modules import CasualMultiheadSelfAttention as cmsa
from modules import TransformerBlock, TransformerLM
from optimizers import SGD, AdamW

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    targets = targets.unsqueeze(-1)
    logits_for_targets = torch.gather(inputs, dim=1, index=targets)
    max_logits, _ = torch.max(inputs, dim=1, keepdim=True)
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs - max_logits), dim=1, keepdim=True))
    log_probs = logits_for_targets - max_logits - log_sum_exp
    loss = -torch.mean(log_probs)

    return loss

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    total_norm = 0.0
    eps = 1e-6

    for p in parameters:
        if p.grad is not None:
            p_norm = p.grad.data.norm(2)
            total_norm += p_norm.item() ** 2

    total_norm = total_norm ** 0.5
    clip_coeff = max_l2_norm / (total_norm + eps)

    if total_norm > max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coeff)
    
    return None

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """

    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)

    elif it <= cosine_cycle_iters:
        cosine_term = math.cos((it - warmup_iters)/(cosine_cycle_iters - warmup_iters) * math.pi)
        decayed_rate = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + cosine_term)
        return decayed_rate

    else:
        return min_learning_rate


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }

    if isinstance(out, (str, os.PathLike)):
        with open(out, 'wb') as f:
            torch.save(ckpt, f)
    else:
        torch.save(ckpt, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    if isinstance(src, (str, os.PathLike)):
        with open(src, 'rb') as f:
            ckpt = torch.load(f)
    else:
        ckpt = torch.load(src)

    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['iteration']

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """

    start_pos = torch.randint(
        low=0,
        high=len(dataset) - context_length,
        size=(batch_size,)
    )
    
    x_batch = [torch.from_numpy(dataset[start : start+context_length]) for start in start_pos]
    y_batch = [torch.from_numpy(dataset[start+1 : start+context_length+1]) for start in start_pos]
    X = torch.stack(x_batch).to(device, dtype=torch.long)
    Y = torch.stack(y_batch).to(device, dtype=torch.long)

    return X, Y