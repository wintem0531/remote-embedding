#!/bin/bash

# 批处理版本启动脚本

echo "========================================"
echo "启动批处理优化版 Embedding 服务"
echo "========================================"

# 激活虚拟环境并启动服务
uv run uvicorn main_batched:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info

echo ""
echo "服务已停止"
