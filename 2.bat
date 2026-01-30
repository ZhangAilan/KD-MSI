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
REM 2. 执行 levir.bat
REM ===============================
echo [INFO] Running levir.bat...
call levir.bat

echo [INFO] All tasks completed.
pause
