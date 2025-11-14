#!/bin/bash
#SBATCH --job-name=ATTN_1e-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=my_job_output_%j.log

TRAIN_DATA_PATH=/mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw1_bpe_and_lm/data/TinyStoriesV2-GPT4-train.npy
VAL_DATA_PATH=/mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw1_bpe_and_lm/data/TinyStoriesV2-GPT4-valid.npy
VOCAB_SIZE=10000
CKPT_PATH=/mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw1_bpe_and_lm/ckpt

echo "Activating virtual environment..."
source /mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw1_bpe_and_lm/.venv/bin/activate

echo "Running Python script..."
python /mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw1_bpe_and_lm/basics/train.py \
  --train_data $TRAIN_DATA_PATH \
  --val_data $VAL_DATA_PATH \
  --vocab_size $VOCAB_SIZE \
  --ckpt_path $CKPT_PATH \
  \
  --model_type "transformer" \
  --context_length 512 \
  --d_model 1024 \
  --num_layers 8 \
  --num_heads 8 \
  --d_ff 4096 \
  --rope_theta 10000.0 \
  \
  --batch_size 64 \
  --max_steps 50000 \
  --max_lr 1e-3 \
  --min_lr 6e-5 \
  --warmup_steps 2000 \
  --weight_decay 0.1 \
  --grad_clip_norm 1.0 \
  \
  --log_interval 100 \
  --ckpt_interval 5000 \
  --device "cuda" \
  --swanlab_project "NLPDL-Transformer" # Set to "None" to disable