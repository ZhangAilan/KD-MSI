#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export DATA_DIR="/home/zyh/code/KD-MSI/Data/whu_CDC_dataset_converted"
export EXPERIMENT_TAG="WHU_KD_T_minus_S_cat_CLIP" # 统一的实验标签
export PREDICT_FOLDER="./experiments/predictions/${EXPERIMENT_TAG}@train@scale=0.5,1.0,1.25,2.0"

echo "=========================================="
echo "Change Detection Pipeline Runner"
echo "=========================================="
echo "1  Train classification with KD"
echo "2  Multi-scale sigmoid inference"
echo "3  Generate pseudo labels"
echo "=========================================="
echo

read -p "Please input start step (1-3): " START_STEP

# 检查输入是否为空或无效
if [[ -z "$START_STEP" ]] || [[ $START_STEP -lt 1 ]] || [[ $START_STEP -gt 3 ]]; then
    echo "Invalid step. Please enter a number between 1 and 3."
    exit 1
fi

echo
echo "Start running from step $START_STEP"
echo

# 根据起始步骤执行相应的命令
if [[ $START_STEP -le 1 ]]; then
    echo "[Step 1] Train classification with KD"
    python train_classification_with_KD_CLIP.py --data_dir "$DATA_DIR" --tag "$EXPERIMENT_TAG" --teacher minus --student cat
    # 检查上一步是否成功
    if [[ $? -ne 0 ]]; then
        echo "Step 1 failed. Exiting."
        exit 1
    fi
fi

if [[ $START_STEP -le 2 ]]; then
    echo "[Step 2] Multi-scale sigmoid inference"
    python multi_scale_sigmoid_inference.py --data_dir "$DATA_DIR" --tag "$EXPERIMENT_TAG" --student_combination cat --scales 0.5,1.0,1.25,2.0
    if [[ $? -ne 0 ]]; then
        echo "Step 2 failed. Exiting."
        exit 1
    fi
fi

if [[ $START_STEP -le 3 ]]; then
    echo "[Step 3] Generate pseudo labels"
    python make_pseudo_labels.py --data_dir "$DATA_DIR" --experiment_name "${EXPERIMENT_TAG}@train@scale=0.5,1.0,1.25,2.0"
    if [[ $? -ne 0 ]]; then
        echo "Step 3 failed. Exiting."
        exit 1
    fi
fi

echo
echo "Pipeline finished"
echo