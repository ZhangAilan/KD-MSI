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
REM 2. 询问是否执行 git pull
REM ===============================
set /p do_git_pull="Do you want to run git pull? (y/n): "

if /i "!do_git_pull!"=="y" (
    goto git_pull_section
) else if /i "!do_git_pull!"=="yes" (
    goto git_pull_section
) else (
    echo [INFO] Skipping git pull as per user choice.
    goto run_whu_bat
)

:git_pull_section
REM ===============================
REM 3. git pull（失败则循环）
REM ===============================
:git_pull_loop
echo [INFO] Running git pull...
git pull

IF ERRORLEVEL 1 (
    echo [WARN] git pull failed. Retrying in 2 seconds...
    timeout /t 2 >nul
    goto git_pull_loop
)

echo [INFO] git pull succeeded.

REM ===============================
REM 4. 选择执行 whu.bat 或 levir.bat
REM ===============================
:choose_bat
echo [INFO] Please choose which bat file to run:
echo [INFO]   0 - Run whu.bat
echo [INFO]   1 - Run levir.bat
set /p bat_choice="Enter your choice (0/1): "

if "!bat_choice!"=="0" (
    echo [INFO] Running whu.bat...
    call whu.bat
) else if "!bat_choice!"=="1" (
    echo [INFO] Running levir.bat...
    call levir.bat
) else (
    echo [ERROR] Invalid choice. Please enter 0 or 1.
    goto choose_bat
)

echo [INFO] All tasks completed.
pause
