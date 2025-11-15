#!/bin/bash
VOCAB_PATH=/mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw1_bpe_and_lm/data/trained_vocab.json
MERGES_PATH=/mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw1_bpe_and_lm/data/trained_merges.txt
VOCAB_SIZE=10000
CKPT_PATH=/mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw1_bpe_and_lm/ckpt/final_transformer.pt
PROMPT="Once upon a time, there was a prince fell in love with his handsome and wedded guard,"
echo "Activating virtual environment..."
source /mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw1_bpe_and_lm/.venv/bin/activate

echo "Running Python script..."
python /mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw1_bpe_and_lm/basics/generate.py \
  --ckpt_path $CKPT_PATH \
  --vocab $VOCAB_PATH \
  --merges $MERGES_PATH \
  --prompt "Once upon a time, there was a prince fell in love with his handsome and wedded guard," \
  --max_gen_len 512 \
  --temperature 0.0 \
  --top_p 0.9 \
  --model_type "transformer" \
  --vocab_size $VOCAB_SIZE \
  --context_length 512 \
  --d_model 1024 \
  --num_layers 8 \
  --num_heads 8 \
  --d_ff 4096