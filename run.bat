@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
REM Anime Image Curator - Windows launcher
REM Requires: uv (https://github.com/astral-sh/uv)
REM
REM   run.bat                      默认：含美观度模型(4个可切换) + WebUI
REM   run.bat --gpu                同上，torch 用 NVIDIA CUDA 版（有 N 卡强烈推荐）
REM   run.bat --base               仅基础（秒开，不装 torch，无美观度模型）
REM   run.bat --folder "D:\imgs"   启动后自动扫描某目录（可组合）
REM   run.bat --port 8080          自定义端口

cd /d "%~dp0"

set UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
set UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
set HF_ENDPOINT=https://hf-mirror.com

where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] 未找到 uv。请先安装：https://github.com/astral-sh/uv
    pause
    exit /b 1
)

set SYNC_MODE=aesthetic
set APP_ARGS=
for %%a in (%*) do (
    set "a=%%~a"
    if /i "!a!"=="--gpu" (
        set SYNC_MODE=gpu
    ) else if /i "!a!"=="--base" (
        set SYNC_MODE=base
    ) else (
        set "APP_ARGS=!APP_ARGS! %%a"
    )
)

if "%SYNC_MODE%"=="base" (
    echo [INFO] 基础模式：同步基础依赖...
    uv sync
) else if "%SYNC_MODE%"=="gpu" (
    echo [INFO] GPU 模式：先装 CUDA 版 torch（首次较慢）...
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 (
        echo [WARN] CUDA 版 torch 安装失败，回退到 CPU 版。
        uv sync --extra aesthetic
    ) else (
        uv sync --extra aesthetic --inexact
    )
) else (
    echo [INFO] 安装/同步 aesthetic 依赖（torch CPU 版；想要 GPU 加速请用 run.bat --gpu）...
    uv sync --extra aesthetic
)
if errorlevel 1 (
    echo.
    echo [ERROR] 依赖安装失败，请看上方报错。
    pause
    exit /b 1
)

echo [INFO] 启动 Anime Image Curator ...
echo [INFO] 浏览器打开 http://127.0.0.1:7861
uv run python -m curator.app !APP_ARGS!

echo.
echo [INFO] Anime Image Curator 已停止。
if errorlevel 1 echo [ERROR] 异常退出，请看上方报错。
pause
