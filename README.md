#!/usr/bin/env bash
# Run the 4f simulation with the specified parameters.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

## Round 1
ROUND="0"
RAW_INPUT_DIR="/cpfs01/projects-HDD/cfff-139269c29e92_HDD/gsb_24110190023/wu/datasets/20260421_data/input_random_20k_sigma_0_5_k_5"
RAW_TARGET_DIR="/cpfs01/projects-HDD/cfff-139269c29e92_HDD/public/AI_optics/Data_store/20260427"
DATA_FOLDER="/cpfs01/projects-HDD/cfff-139269c29e92_HDD/gsb_24110190023/wu/AIOptics/data/fiber_random_20k_exp"
NEWTON_SAVE_DIR="fiber_random_20k_emnist_20k_exp"
INPUT_MODE="phase"
CHECKPOINT_PATH="checkpoints/fiber_random_20k_exp"
EXP_NAME="fiber_random_20k_exp"
INPUT_SIZE=50
PAT_SIZE=100
INPUT_CHANNELS=2
INPUT_SCALE="2_pi"
CLIP_SPECKLE=65535

# Data preprocess
python datasets/parallel_preprocess_input_simulation.py \
    --raw_input_folder "${RAW_INPUT_DIR}" \
    --raw_target_folder "${RAW_TARGET_DIR}" \
    --save_folder "${DATA_FOLDER}" \
    --input-mode "${INPUT_MODE}"

# Training 
python main_stage1.py \
    --root_dir "${DATA_FOLDER}" \
    --batch_size 32 \
    --num_epochs 400 \
    --lr 4e-4 \
    --weight_decay 0.05 \
    --gpu_ids 0 \
    --save_path "${CHECKPOINT_PATH}" \
    --input_size "${INPUT_SIZE}" \
    --pat_size "${PAT_SIZE}" \
    --exp_name "${EXP_NAME}_round${ROUND}" \
    --input_scale "${INPUT_SCALE}" \
    --clip_speckle ${CLIP_SPECKLE} \
    --input_channels "${INPUT_CHANNELS}"

# Inference
python parallel_inference_newton.py \
    --batch-size 32 \
    --num-steps 200 \
    --input-size "${INPUT_SIZE}" \
    --pat-size "${PAT_SIZE}" \
    --mode "${NEWTON_SAVE_DIR}" \
    --checkpoint "${CHECKPOINT_PATH}/stage1/best.pth" \
    --num-samples 20000 \
    --input-channels "${INPUT_CHANNELS}" \
    --input-mode "${INPUT_MODE}" \
    --input-scale "${INPUT_SCALE}"
