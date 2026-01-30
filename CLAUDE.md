# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个用于远程嵌入(remote embedding)的 Python 项目,使用 ModelScope 和 Transformers 库进行文本嵌入处理。

## 依赖管理

项目使用 `uv` 作为包管理器:

```bash
# 安装依赖
uv sync

# 添加新依赖
uv add <package-name>

# 运行 Python 脚本
uv run python main.py
```

## 环境要求

- Python 3.12
- 依赖包:
  - modelscope >= 1.34.0
  - transformers == 4.51.3

## 项目结构

- `main.py`: 主入口文件
- `pyproject.toml`: 项目配置和依赖定义
- `.venv/`: 虚拟环境目录
