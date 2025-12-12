#!/bin/bash
#SBATCH --job-name=HF_EXP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=my_job_output_%j.log


export SWANLAB_PROJECT="NLPDL_HW2_Task2_Training"
export HF_ENDPOINT=https://hf-mirror.com 


# --- Experiment Settings ---
# MODELS=("bert-base-uncased" "facebook/bart-base" "Qwen/Qwen1.5-0.5B")
MODELS=("bert-base-uncased" "facebook/bart-base" "Qwen/Qwen1.5-0.5B")

DATASETS=("restaurant_sup" "agnews_sup" "acl_sup") 

SEEDS=(2020 2021 2022 2023 2024 2025)

LR_FULL=2e-5      # Lower LR for full fine-tuning
LR_PEFT=2e-4      # Higher LR is often better for PEFT
EPOCHS=4

# Base output directory
OUTPUT_BASE="./results"

echo "Activating virtual environment..."
source /mnt/nfs_project_a/yichen/NLPDL-2025Fall/hw2_huggingface/.venv/bin/activate

# --- Main Loop ---
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            
            # Create a safe directory name (replace / with -)
            SAFE_MODEL_NAME=${model/\//-}
            
            echo "========================================================"
            echo "Processing: Model=$model | Dataset=$dataset | Seed=$seed"
            echo "========================================================"

            # ---------------------------------------------------------
            # 1. Full Fine-Tuning (Task 2)
            # ---------------------------------------------------------
            echo "[Task 2] Running Full Fine-Tuning..."
            python train.py \
                --model_name "$model" \
                --dataset "$dataset" \
                --output_dir "$OUTPUT_BASE/$SAFE_MODEL_NAME/$dataset/full/seed_$seed" \
                --do_train \
                --do_eval \
                --num_train_epochs $EPOCHS \
                --learning_rate $LR_FULL \
                --per_device_train_batch_size 16 \
                --eval_strategy "epoch" \
                --save_strategy "epoch" \
                --load_best_model_at_end \
                --metric_for_best_model "accuracy" \
                --seed "$seed" \
                --overwrite_output_dir \
                --trust_remote_code True \
                --report_to "none"

        done
    done
done

echo "All experiments finished successfully."
