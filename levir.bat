@echo off
setlocal enabledelayedexpansion

set CUDA_VISIBLE_DEVICES=0
set DATA_DIR=E:\weakly_CD_dataset\dataset\Levir_CDC_dataset\Levir_CDC_dataset_converted
set EXPERIMENT_TAG=LEVIR_KD_T_abs_S_cat

echo ==========================================
echo Change Detection Pipeline Runner
echo ==========================================
echo 1  Train classification with KD
echo 2  Multi-scale sigmoid inference
echo 3  Generate pseudo labels
echo ==========================================
echo.

set /p START_STEP=Please input start step (1-3):

echo.
echo Select domain:
echo 0 - train
echo 1 - test
echo.
set /p DOMAIN_CHOICE=Please input domain choice (0/1):

if "%DOMAIN_CHOICE%"=="0" (
    set DOMAIN=train
) else (
    set DOMAIN=test
)

if "%START_STEP%"=="" goto END
if %START_STEP% GTR 3 goto END
if %START_STEP% LSS 1 goto END

echo.
echo Start running from step %START_STEP%
echo.

if %START_STEP% LEQ 1 (
    echo [Step 1] Train classification with KD
    python train_classification_with_KD_CLIP.py --data_dir %DATA_DIR% --tag %EXPERIMENT_TAG% --teacher minus --student cat
)

if %START_STEP% LEQ 2 (
    echo [Step 2] Multi-scale sigmoid inference
    python multi_scale_sigmoid_inference.py --data_dir %DATA_DIR% --tag %EXPERIMENT_TAG% --student_combination cat --scales 0.5,1.0,1.25,2.0 --domain %DOMAIN%
)

if %START_STEP% LEQ 3 (
    echo [Step 3] Generate pseudo labels
    python make_pseudo_labels.py --data_dir %DATA_DIR% --experiment_name %EXPERIMENT_TAG% --domain %DOMAIN%
)

echo.
echo Pipeline finished
echo.

:END
pause
