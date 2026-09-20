#!/usr/bin/env bash
# Anime Image Curator - Linux/WSL launcher
# Requires: uv (https://github.com/astral-sh/uv)
#
#   ./run.sh                        默认：含美观度模型(4个可切换) + WebUI
#   ./run.sh --base                 仅基础（秒开，不装 torch，无美观度模型）
#   ./run.sh --folder /path/to/imgs 启动后自动扫描
#   ./run.sh --port 8080            自定义端口
#
# 注：本机(WSL 网关)无 GPU；如需 CUDA 版 torch，手动：
#   uv pip install torch --index-url https://download.pytorch.org/whl/cu124
set -e
cd "$(dirname "$0")"

export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com

command -v uv >/dev/null 2>&1 || { echo "[ERROR] 未找到 uv。安装：https://github.com/astral-sh/uv"; exit 1; }

if [[ " $* " == *" --base "* ]]; then
    echo "[INFO] 基础模式：同步基础依赖..."
    uv sync
else
    echo "[INFO] 安装/同步 aesthetic 依赖..."
    uv sync --extra aesthetic
fi

echo "[INFO] 启动 Anime Image Curator -> http://127.0.0.1:7861"
uv run python -m curator.app "$@"
