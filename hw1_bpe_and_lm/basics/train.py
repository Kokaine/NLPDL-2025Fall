import os
import time
import math
import argparse
import numpy as np
import torch
from torch.nn.parameter import Parameter
from typing import List, Optional
from tqdm import tqdm

from modules import TransformerLM, LSTMLM
from optimizers import AdamW, SGD
from utils import (
    cross_entropy,
    get_lr_cosine_schedule,
    gradient_clipping,
    get_batch,
    load_checkpoint,
    save_checkpoint
)

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False

# --- Configuration & Argument Parsing ---
def get_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Language Model")
    
    # --- I/O and Checkpointing ---
    parser.add_argument(
        "--train_data", type=str, required=True, 
        help="Path to the memory-mapped .npy file for training data."
    )
    parser.add_argument(
        "--val_data", type=str, required=True, 
        help="Path to the memory-mapped .npy file for validation data."
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="model.pt", 
        help="Path to save/load checkpoints."
    )
    
    # --- Model Architecture ---
    parser.add_argument(
        "--model_type", type=str, default="transformer", 
        choices=["transformer", "lstm"], help="Model type to train."
    )
    parser.add_argument(
        "--vocab_size", type=int, required=True, 
        help="Size of the vocabulary."
    )
    parser.add_argument(
        "--context_length", type=int, default=256, 
        help="Sequence length for training."
    )
    parser.add_argument(
        "--d_model", type=int, default=512, 
        help="Model hidden dimension."
    )
    parser.add_argument(
        "--num_layers", type=int, default=8, 
        help="Number of layers in the model."
    )
    # Transformer
    parser.add_argument(
        "--num_heads", type=int, default=8, 
        help="Number of attention heads (for Transformer)."
    )
    parser.add_argument(
        "--d_ff", type=int, default=2048, 
        help="Dimension of the FFN (for Transformer)."
    )
    parser.add_argument(
        "--rope_theta", type=float, default=10000.0, 
        help="RoPE theta parameter (for Transformer)."
    )

    # --- Training & Optimization ---
    parser.add_argument(
        "--batch_size", type=int, default=64, 
        help="Batch size for training."
    )
    parser.add_argument(
        "--max_steps", type=int, default=500000, 
        help="Total number of training steps."
    )
    parser.add_argument(
        "--max_lr", type=float, default=6e-4, 
        help="Maximum learning rate (alpha_max)."
    )
    parser.add_argument(
        "--min_lr", type=float, default=6e-5, 
        help="Minimum learning rate (alpha_min)."
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=2000, 
        help="Number of linear warmup steps (T_w)."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, 
        help="AdamW weight decay (lambda)."
    )
    parser.add_argument(
        "--beta1", type=float, default=0.9, 
        help="AdamW beta1."
    )
    parser.add_argument(
        "--beta2", type=float, default=0.95, 
        help="AdamW beta2 (LLaMA uses 0.95)."
    )
    parser.add_argument(
        "--grad_clip_norm", type=float, default=1.0, 
        help="Max norm for gradient clipping."
    )

    # --- Logging and System ---
    parser.add_argument(
        "--log_interval", type=int, default=100, 
        help="Log training stats every N steps."
    )
    parser.add_argument(
        "--ckpt_interval", type=int, default=1000, 
        help="Save a checkpoint every N steps."
    )
    parser.add_argument(
        "--device", type=str, default="auto", 
        help="Device to use ('cpu', 'cuda', 'auto')."
    )
    parser.add_argument(
        "--swanlab_project", type=str, default=None, 
        help="Weights & Biases project name. If None, swanlab is disabled."
    )
    parser.add_argument(
        "--ckpt_resume", type=bool, default=False, 
        help="Resuming from a previous training."
    )

    return parser.parse_args()

def setup_device(device_str: str) -> str:
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_str

def load_data(path: str) -> np.ndarray:
    print(f"Loading data from {path} (memory-mapped)...")
    try:
        data = np.load(path, mmap_mode='r', allow_pickle=True)
        return data
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {path}")
        exit(1)
    except Exception as e:
        print(f"ERROR: Could not load data from {path}: {e}")
        exit(1)

def setup_model(args: argparse.Namespace) -> torch.nn.Module:
    print(f"Initializing {args.model_type} model...")
    if args.model_type == "transformer":
        model = TransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            use_rope=True,
            theta=args.rope_theta,
            device=args.device,
            dtype=torch.float32
        )
    elif args.model_type == "lstm":
        model = LSTMLM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            device=args.device,
            dtype=torch.float32
        )
    return model.to(args.device)

def setup_logging(args: argparse.Namespace):
    if args.swanlab_project and SWANLAB_AVAILABLE:
        print("Initializing Weights & Biases...")
        swanlab.init(project=args.swanlab_project, config=vars(args))
        return True
    elif args.swanlab_project:
        print("Warning: --swanlab_project was specified, but swanlab is not installed.")
    return False

@torch.no_grad()
def evaluate(
    model: torch.nn.Module, 
    val_data: np.ndarray, 
    args: argparse.Namespace
) -> float:
    model.eval()
    
    val_x, val_y = get_batch(
        val_data, 
        args.batch_size, 
        args.context_length, 
        args.device
    )

    logits = model(val_x)
    logits_for_loss = logits.permute(0, 2, 1)
    val_loss = cross_entropy(logits_for_loss, val_y)
    
    model.train()
    return val_loss.item()

def main():
    args = get_args()
    args.device = setup_device(args.device)
    print(f"Using device: {args.device}")
    
    torch.manual_seed(1337)
    
    use_swanlab = setup_logging(args)
    train_data = load_data(args.train_data)
    val_data = load_data(args.val_data)
    
    model = setup_model(args)
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    if args.ckpt_resume:
        start_step = load_checkpoint(os.path.join(args.ckpt_path, f"latest_{args.model_type}.pt"), model, optimizer, args.device)
        print(f"Starting training from step {start_step+1}...")
    else:
        start_step = 0

    model.train()
    start_time = time.time()

    pbar = tqdm(
        range(start_step, args.max_steps),
        initial=start_step,
        total=args.max_steps,
        desc="Training"
    )

    for step in pbar:
        current_step_num = step + 1
        
        lr = get_lr_cosine_schedule(
            it=current_step_num,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_steps,
            cosine_cycle_iters=args.max_steps
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        x, y = get_batch(
            train_data, 
            args.batch_size, 
            args.context_length, 
            args.device
        )
        
        logits = model(x)
        logits_for_loss = logits.permute(0, 2, 1)
        loss = cross_entropy(logits_for_loss, y)

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm=args.grad_clip_norm)
        optimizer.step()
        
        # Logging
        if current_step_num % args.log_interval == 0:
            val_loss = evaluate(model, val_data, args)
            end_time = time.time()
            time_per_step = (end_time - start_time) * 1000 / args.log_interval
            print(
                f"Step {current_step_num}/{args.max_steps} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time/Step: {time_per_step:.2f}ms"
            )
            
            if use_swanlab:
                swanlab.log({
                    "step": current_step_num,
                    "train_loss": loss.item(),
                    "val_loss": val_loss,
                    "lr": lr,
                    "time_per_step_ms": time_per_step
                })
            
            start_time = time.time()

        if current_step_num % args.ckpt_interval == 0:
            if current_step_num - args.ckpt_interval > 0:
                old_path = os.path.join(args.ckpt_path, f"latest_{args.model_type}.pt")
                if os.path.exists(old_path):
                    new_path = os.path.join(args.ckpt_path, f"{current_step_num - args.ckpt_interval}_{args.model_type}.pt")
                    os.rename(old_path, new_path)
            save_checkpoint(
                model, 
                optimizer, 
                current_step_num,
                os.path.join(args.ckpt_path, f"latest_{args.model_type}.pt")
            )
            
    print("Training finished.")
    save_checkpoint(
        model, 
        optimizer, 
        args.max_steps,
        os.path.join(args.ckpt_path, f"final_{args.model_type}.pt")
    )

if __name__ == "__main__":
    main()