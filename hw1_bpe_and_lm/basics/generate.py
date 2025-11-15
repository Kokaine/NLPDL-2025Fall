import argparse
import torch
import numpy as np

# Import all your custom modules
from modules import TransformerLM, LSTMLM
from tokenizer import Tokenizer # Assumes your tokenizer is in tokenizer.py

def get_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained model")
    
    # --- Model & Tokenizer Paths ---
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the .pt checkpoint file.")
    parser.add_argument("--vocab", type=str, required=True, help="Path to vocab.json.")
    parser.add_argument("--merges", type=str, required=True, help="Path to merges.txt.")
    
    # --- Generation Parameters ---
    parser.add_argument("--prompt", type=str, default="Once upon a time,")
    parser.add_argument("--max_gen_len", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--special_tokens", type=str, default="<|endoftext|>")

    # --- Model Architecture (MUST match the checkpoint) ---
    parser.add_argument("--model_type", type=str, required=True, choices=["transformer", "lstm"])
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()

def main():
    args = get_args()
    
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        args.vocab, 
        args.merges, 
        special_tokens=[args.special_tokens]
    )
    special_token_id = tokenizer.token_encode[args.special_tokens.encode('utf-8')]

    print("Initializing model...")
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
    else: # lstm
        model = LSTMLM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            device=args.device,
            dtype=torch.float32
        )
    
    model.to(args.device)

    # print(f"Loading checkpoint from {args.ckpt_path}...")
    # We only need the model's state_dict, not the optimizer's
    ckpt = torch.load(args.ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt['model_state_dict'])

    print(f"Encoding prompt: '{args.prompt}'")
    prompt_ids = tokenizer.encode(args.prompt)
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=args.device).unsqueeze(0)

    print("--- Generating ---")
    
    # Call the model's new .generate() method
    generated_ids = model.generate(
        prompt_tensor,
        args.max_gen_len,
        args.temperature,
        args.top_p,
        special_token_id
    )

    # Decode the output
    generated_text = tokenizer.decode(generated_ids)
    
    print(generated_text)
    print("--------------------")

if __name__ == "__main__":
    main()