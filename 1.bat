@echo off
setlocal enabledelayedexpansion

REM ===============================
REM 1. 激活 conda 环境
REM ===============================
call conda activate zyh_wscd
IF ERRORLEVEL 1 (
    echo [ERROR] Failed to activate conda environment: zyh_wscd
    pause
    exit /b 1
)

echo [INFO] Conda environment activated.

REM ===============================
REM 2. git pull（失败则循环）
REM ===============================
:git_pull_loop
echo [INFO] Running git pull...
git pull

IF ERRORLEVEL 1 (
    echo [WARN] git pull failed. Retrying in 5 seconds...
    timeout /t 5 >nul
    goto git_pull_loop
)

echo [INFO] git pull succeeded.

REM ===============================
REM 3. 执行 whu.bat
REM ===============================
echo [INFO] Running whu.bat...
call whu.bat

echo [INFO] All tasks completed.
pause
