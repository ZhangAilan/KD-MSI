@echo off
setlocal enabledelayedexpansion

set CUDA_VISIBLE_DEVICES=0
set DATA_DIR=E:\weakly_CD_dataset\WHU-CD-256\WHU-CD-256
set PREDICT_FOLDER=predict_folder

echo ==========================================
echo Change Detection Pipeline Runner
echo ==========================================
echo 1  Train classification with KD
echo 2  Multi-scale sigmoid inference
echo 3  Evaluate change probability map
echo 4  Generate pseudo labels
echo 5  Train change detection model
echo 6  Inference change detection
echo ==========================================
echo.

set /p START_STEP=Please input start step (1-6): 

if "%START_STEP%"=="" goto END
if %START_STEP% GTR 6 goto END
if %START_STEP% LSS 1 goto END

echo.
echo Start running from step %START_STEP%
echo.

if %START_STEP% LEQ 1 (
    echo [Step 1] Train classification with KD
    python train_classification_with_KD.py --data_dir %DATA_DIR% --tag WHU_KD_T_minus_S_cat --teacher minus --student cat
)

if %START_STEP% LEQ 2 (
    echo [Step 2] Multi-scale sigmoid inference
    python multi_scale_sigmoid_inference.py --data_dir %DATA_DIR% --tag WHU_KD_T_minus_S_cat --student_combination minus --scales 0.5,1.0,1.25,2.0
)

if %START_STEP% LEQ 3 (
    echo [Step 3] Evaluate change probability map
    python evaluate.py --list_file train.txt --predict_folder %PREDICT_FOLDER% --mode npy --data_dir %DATA_DIR%
)

if %START_STEP% LEQ 4 (
    echo [Step 4] Generate pseudo labels
    python make_pseudo_labels.py --data_dir %DATA_DIR% --experiment_name WHU_KD_T_minus_S_cat@train@scale=0.5,1.0,1.25,2.0 --domain train --threshold 0.3
)

if %START_STEP% LEQ 5 (
    echo [Step 5] Train change detection model
    python train_change_detection.py --data_dir %DATA_DIR% --tag WHU_weakly_change_detection --label_name WHU_KD_T_minus_S_cat@train@scale=0.5,1.0,1.25,2.0@crf=0@255@threshold0.3
)

if %START_STEP% LEQ 6 (
    echo [Step 6] Inference change detection
    python inference_change_detection.py --data_dir %DATA_DIR% --tag WHU_weakly_change_detection --scales 0.5,1.0,1.5,2.0
)

echo.
echo Pipeline finished
echo.

:END
pause
