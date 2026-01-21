@echo off
setlocal enabledelayedexpansion

set CUDA_VISIBLE_DEVICES=0
set DATA_DIR=E:\weakly_CD_dataset\WHU-CD-256\WHU-CD-256
set PREDICT_FOLDER=predict_folder

set START_STEP=
set END_STEP=

if "%1"=="" (
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
    set /p END_STEP=Please input end step (1-6):
) else (
    set START_STEP=%1
    set END_STEP=%2
)

REM ======== 安全兜底（关键）========
if "%START_STEP%"=="" set START_STEP=1
if "%END_STEP%"=="" set END_STEP=6

REM ======== 合法性修正 ========
if %START_STEP% LSS 1 set START_STEP=1
if %END_STEP% GTR 6 set END_STEP=6

if %START_STEP% GTR %END_STEP% (
    echo Invalid range: START_STEP ^> END_STEP
    goto END
)

echo.
echo Running steps from %START_STEP% to %END_STEP%
echo ==========================================
echo.

if %START_STEP% LEQ 1 if %END_STEP% GEQ 1 (
    echo [Step 1] Train classification with KD
    python train_classification_with_KD.py --data_dir %DATA_DIR% --tag WHU_KD_T_minus_S_cat --teacher minus --student cat
    if errorlevel 1 goto END
)

if %START_STEP% LEQ 2 if %END_STEP% GEQ 2 (
    echo [Step 2] Multi-scale sigmoid inference
    python multi_scale_sigmoid_inference.py --data_dir %DATA_DIR% --tag WHU_KD_T_minus_S_cat --student_combination minus --scales 0.5,1.0,1.25,2.0
    if errorlevel 1 goto END
)

if %START_STEP% LEQ 3 if %END_STEP% GEQ 3 (
    echo [Step 3] Evaluate change probability map
    python evaluate.py --list_file train.txt --predict_folder %PREDICT_FOLDER% --mode npy --data_dir %DATA_DIR%
    if errorlevel 1 goto END
)

if %START_STEP% LEQ 4 if %END_STEP% GEQ 4 (
    echo [Step 4] Generate pseudo labels
    python make_pseudo_labels.py --data_dir %DATA_DIR% --experiment_name WHU_KD_T_minus_S_cat@train@scale=0.5,1.0,1.25,2.0 --domain train --threshold 0.3
    if errorlevel 1 goto END
)

if %START_STEP% LEQ 5 if %END_STEP% GEQ 5 (
    echo [Step 5] Train change detection model
    python train_change_detection.py --data_dir %DATA_DIR% --tag WHU_weakly_change_detection --label_name WHU_KD_T_minus_S_cat@train@scale=0.5,1.0,1.25,2.0@crf=0@255@threshold0.3
    if errorlevel 1 goto END
)

if %START_STEP% LEQ 6 if %END_STEP% GEQ 6 (
    echo [Step 6] Inference change detection
    python inference_change_detection.py --data_dir %DATA_DIR% --tag WHU_weakly_change_detection --scales 0.5,1.0,1.5,2.0
    if errorlevel 1 goto END
)

:END
echo.
echo ==========================================
echo Pipeline finished
echo ==========================================
pause
