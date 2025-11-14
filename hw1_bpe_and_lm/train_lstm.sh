#!/bin/bash

TRAIN_DATA_PATH=/mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw1_bpe_and_lm/data/TinyStoriesV2-GPT4-train.txt
VAL_DATA_PATH=/mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw1_bpe_and_lm/data/TinyStoriesV2-GPT4-valid.txt
VOCAB_SIZE=10000
CKPT_PATH=/mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw1_bpe_and_lm/ckpt

echo "Activating virtual environment..."
source /mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw1_bpe_and_lm/.venv/bin/activate

# Run your Python script
echo "Running Python script..."
python /mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw1_bpe_and_lm/basics/train.py \
    --train_data $TRAIN_DATA_PATH \
    --val_data $VAL_DATA_PATH \
    --vocab_size $VOCAB_SIZE \
    --ckpt_path $CKPT_PATH \
    \
    --model_type "lstm" \
    --context_length 256 \
    --d_model 512 \
    --num_layers 8 \
    \
    --batch_size 64 \
    --max_steps 500000 \
    --max_lr 6e-4 \
    --min_lr 6e-5 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --grad_clip_norm 1.0 \
    \
    --log_interval 100 \
    --ckpt_interval 1000 \
    --device "auto" \
    --swanlab_project "NLPDL-LSTM"

# Deactivation is optional, as the job will end anyway
echo "Job finished."