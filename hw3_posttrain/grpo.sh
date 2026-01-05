#!/bin/bash

#SBATCH --partition=h100
#SBATCH --job-name=GRPO
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --time 24:00:00

# --- Setup Environment ---

# 1. Navigate to your project directory
PROJECT_DIR="/mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw3_posttrain"
cd "$PROJECT_DIR" || exit


# 3. Load Modules (Adjust based on your cluster, e.g., via 'module avail')
# module load cuda/12.1
# module load python/3.10

# 4. Set Environment Variables for vLLM & HuggingFace
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=
# --- Run Script ---

echo "Job started on $(hostname) at $(date)"
echo "Running in directory: $(pwd)"

# Execute using uv
# 'uv run' will automatically use the virtual environment managed by uv
uv run grpo.py

echo "Job finished at $(date)"