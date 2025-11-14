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


# Weights and Biases (wandb) for logging (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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
    # Transformer-specific
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

    # --- Logging & System ---
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
        "--wandb_project", type=str, default=None, 
        help="Weights & Biases project name. If None, wandb is disabled."
    )

    return parser.parse_args()

# --- Helper Functions ---

def setup_device(device_str: str) -> str:
    """Selects the best available device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_str

def load_data(path: str) -> np.ndarray:
    """Loads a .npy file in memory-mapped mode."""
    print(f"Loading data from {path} (memory-mapped)...")
    try:
        data = np.load(path, mmap_mode='r')
        return data
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {path}")
        exit(1)
    except Exception as e:
        print(f"ERROR: Could not load data from {path}: {e}")
        exit(1)

def setup_model(args: argparse.Namespace) -> torch.nn.Module:
    """Initializes the correct model based on args."""
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
            dtype=torch.float32  # AdamW works best with float32
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
    """Initializes Weights & Biases if requested."""
    if args.wandb_project and WANDB_AVAILABLE:
        print("Initializing Weights & Biases...")
        wandb.init(project=args.wandb_project, config=vars(args))
        return True
    elif args.wandb_project:
        print("Warning: --wandb_project was specified, but wandb is not installed.")
    return False

@torch.no_grad()
def evaluate(
    model: torch.nn.Module, 
    val_data: np.ndarray, 
    args: argparse.Namespace
) -> float:
    """Runs one evaluation step and returns the loss."""
    model.eval()
    
    val_x, val_y = run_get_batch(
        val_data, 
        args.batch_size, 
        args.context_length, 
        args.device
    )
    
    # Get logits
    if args.model_type == "transformer":
        pos = torch.arange(
            0, args.context_length, device=args.device
        ).unsqueeze(0) # Shape (1, context_length)
        logits = model(val_x, pos)
    else: # lstm
        logits = model(val_x)
        
    val_loss = run_cross_entropy(logits, val_y)
    
    model.train()
    return val_loss.item()

# --- Main Training Loop ---

def main():
    args = get_args()
    
    # --- Setup ---
    args.device = setup_device(args.device)
    print(f"Using device: {args.device}")
    
    torch.manual_seed(1337) # for reproducibility
    
    use_wandb = setup_logging(args)
    
    # --- Data ---
    train_data = load_data(args.train_data)
    val_data = load_data(args.val_data)
    
    # --- Model & Optimizer ---
    model = setup_model(args)
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr, # Initial LR is max, scheduler will adjust
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    # --- Checkpoint Resuming ---
    start_step = load_checkpoint(args.ckpt_path, model, optimizer, args.device)
    
    print(f"Starting training from step {start_step+1}...")
    model.train()
    start_time = time.time()
    
    # --- Training Loop ---
    for step in range(start_step, args.max_steps):
        current_step_num = step + 1 # Use 1-based indexing for scheduler
        
        # 1. Update Learning Rate
        lr = get_lr_cosine_schedule(
            t=current_step_num,
            alpha_max=args.max_lr,
            alpha_min=args.min_lr,
            T_w=args.warmup_steps,
            T_c=args.max_steps
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 2. Get Data Batch
        x, y = run_get_batch(
            train_data, 
            args.batch_size, 
            args.context_length, 
            args.device
        )
        
        # 3. Forward Pass
        if args.model_type == "transformer":
            pos = torch.arange(
                0, args.context_length, device=args.device
            ).unsqueeze(0) # Shape (1, context_length)
            logits = model(x, pos)
        else: # lstm
            # LSTM forward pass does not take position_ids
            # and manages its own state internally
            logits = model(x)
            
        loss = run_cross_entropy(logits, y)

        # 4. Backward Pass
        optimizer.zero_grad()
        loss.backward()
        
        # 5. Gradient Clipping
        gradient_clipping(model.parameters(), max_norm=args.grad_clip_norm)
        
        # 6. Optimizer Step
        optimizer.step()
        
        # 7. Logging
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
            
            if use_wandb:
                wandb.log({
                    "step": current_step_num,
                    "train_loss": loss.item(),
                    "val_loss": val_loss,
                    "lr": lr,
                    "time_per_step_ms": time_per_step
                })
            
            start_time = time.time() # Reset timer
            
        # 8. Checkpointing
        if current_step_num % args.ckpt_interval == 0:
            save_checkpoint(
                args.ckpt_path, 
                model, 
                optimizer, 
                current_step_num
            )
            
    print("Training finished.")
    save_checkpoint(
        args.ckpt_path.replace(".pt", "_final.pt"), 
        model, 
        optimizer, 
        args.max_steps
    )

if __name__ == "__main__":
    main()