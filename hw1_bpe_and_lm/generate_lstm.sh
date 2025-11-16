#!/bin/bash
VOCAB_PATH=/mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw1_bpe_and_lm/data/trained_vocab.json
MERGES_PATH=/mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw1_bpe_and_lm/data/trained_merges.txt
VOCAB_SIZE=10000
CKPT_PATH=/mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw1_bpe_and_lm/ckpt/final_lstm.pt

echo "Activating virtual environment..."
source /mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw1_bpe_and_lm/.venv/bin/activate

echo "Running Python script..."
python /mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw1_bpe_and_lm/basics/generate.py \
  --ckpt_path $CKPT_PATH \
  --vocab $VOCAB_PATH \
  --merges $MERGES_PATH \
  --prompt "On Tuesday, everyone in the city who owned a cat woke up to find it had been replaced by a small, perfectly carved wooden owl." \
  --max_gen_len 512 \
  --temperature 0.9 \
  --top_p 0.9 \
  --model_type "lstm" \
  --vocab_size $VOCAB_SIZE \
  --context_length 512 \
  --d_model 512 \
  --num_layers 8