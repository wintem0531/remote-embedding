# 批处理版本部署指南

## 文件说明

- `main_batched.py`: 批处理优化版本的主程序
- `test_batch.py`: 性能测试脚本
- `run_batched.sh`: 启动脚本
- `main.py`: 原始版本(保留作为备份)

## 部署步骤

### 1. 上传文件到服务器

将以下文件上传到服务器:
```bash
scp main_batched.py your_server:/path/to/project/
scp test_batch.py your_server:/path/to/project/
scp run_batched.sh your_server:/path/to/project/
scp pyproject.toml your_server:/path/to/project/
```

### 2. 在服务器上安装依赖

```bash
# SSH 登录服务器
ssh your_server

# 进入项目目录
cd /path/to/project/

# 安装 uv (如果还没安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖
uv sync

# 安装测试工具依赖
uv add httpx
```

### 3. 下载模型(可选但推荐)

**方式 A: 使用 ModelScope (国内推荐)**
```bash
uv run python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('iic/gme-Qwen2-VL-7B-Instruct', cache_dir='./models')
print(f'模型已下载到: {model_dir}')
"
```

**方式 B: 使用 Hugging Face**
```bash
uv run huggingface-cli download iic/gme-Qwen2-VL-7B-Instruct --local-dir ./models/gme-Qwen2-VL-7B-Instruct
```

如果下载了本地模型,修改 `main_batched.py:345` 行:
```python
# 将
model_path = "iic/gme-Qwen2-VL-7B-Instruct"

# 改为
model_path = "./models/gme-Qwen2-VL-7B-Instruct"
```

### 4. 调整批处理参数(可选)

编辑 `main_batched.py` 的配置部分 (第 17-22 行):

```python
class BatchConfig:
    """批处理配置"""
    MAX_BATCH_SIZE = 16      # 批次大小,建议 8-32
    MAX_WAIT_MS = 50         # 等待时间(毫秒),建议 20-100
    ENABLE_BATCHING = True   # 是否启用批处理
```

**参数说明**:
- `MAX_BATCH_SIZE`: 越大吞吐量越高,但单次推理延迟增加
  - 低延迟场景: 8-12
  - 高吞吐场景: 16-32

- `MAX_WAIT_MS`: 等待时间越长,批次越满,但延迟增加
  - 低延迟场景: 20-50 ms
  - 高吞吐场景: 50-100 ms

### 5. 启动服务

**方式 A: 使用启动脚本**
```bash
chmod +x run_batched.sh
./run_batched.sh
```

**方式 B: 直接使用 uvicorn**
```bash
uv run uvicorn main_batched:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info
```

**方式 C: 后台运行 (推荐生产环境)**
```bash
# 使用 nohup
nohup uv run uvicorn main_batched:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    > embedding_service.log 2>&1 &

# 查看日志
tail -f embedding_service.log

# 查看进程
ps aux | grep uvicorn

# 停止服务
pkill -f "uvicorn main_batched:app"
```

**方式 D: 使用 systemd (最推荐)**

创建服务文件 `/etc/systemd/system/embedding.service`:
```ini
[Unit]
Description=Remote Embedding API Service
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/project
ExecStart=/home/your_username/.local/bin/uv run uvicorn main_batched:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务:
```bash
sudo systemctl daemon-reload
sudo systemctl enable embedding
sudo systemctl start embedding
sudo systemctl status embedding

# 查看日志
sudo journalctl -u embedding -f
```

### 6. 测试服务

```bash
# 测试健康检查
curl http://localhost:8000/

# 测试文本嵌入
curl -X POST http://localhost:8000/embeddings/text \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["测试文本1", "测试文本2"]
  }'

# 查看统计信息
curl http://localhost:8000/stats

# 运行性能测试 (需要在另一个终端)
uv run python test_batch.py
```

## 性能监控

### 查看实时统计
```bash
# 每 5 秒查看一次统计信息
watch -n 5 'curl -s http://localhost:8000/stats | python -m json.tool'
```

### 查看 GPU 使用情况
```bash
# 实时监控
watch -n 1 nvidia-smi

# 或使用 nvtop (更直观)
nvtop
```

## 性能调优建议

### 1. 内存优化
如果遇到显存不足:
```python
# 在 main_batched.py:345 添加
torch_dtype="float16"  # 或 "bfloat16"
# 或者使用量化
load_in_8bit=True
```

### 2. 批处理参数调优

根据实际负载调整:

| 场景 | MAX_BATCH_SIZE | MAX_WAIT_MS | 预期并发 |
|------|----------------|-------------|----------|
| 在线服务 (低延迟) | 8-12 | 20-30 | 8-10 |
| 离线处理 (高吞吐) | 24-32 | 80-100 | 15-20 |
| 平衡模式 | 16 | 50 | 10-15 |

### 3. 系统级优化
```bash
# 增加文件描述符限制
ulimit -n 65535

# 优化网络参数
echo "net.core.somaxconn = 1024" >> /etc/sysctl.conf
sysctl -p
```

## 故障排查

### 问题 1: 模型加载失败
```bash
# 检查 transformers 版本
uv run python -c "import transformers; print(transformers.__version__)"

# 应该是 4.51.3,如果不是:
uv pip install transformers==4.51.3
```

### 问题 2: CUDA 错误
```bash
# 检查 CUDA 版本
nvidia-smi

# 检查 PyTorch CUDA 支持
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### 问题 3: 端口被占用
```bash
# 查看端口占用
lsof -i :8000

# 更换端口
uv run uvicorn main_batched:app --port 8001
```

## API 接口说明

### 1. 健康检查
```
GET /
返回: 服务状态和配置信息
```

### 2. 统计信息
```
GET /stats
返回: 批处理统计信息和队列状态
```

### 3. 文本嵌入
```
POST /embeddings/text
请求体: {"texts": ["文本1", "文本2"], "instruction": "可选指令"}
返回: {"embeddings": [[...]], "dimension": 1024}
```

### 4. 图像嵌入
```
POST /embeddings/image
请求体: {"images": ["url1", "url2"], "is_query": true}
返回: {"embeddings": [[...]], "dimension": 1024}
```

### 5. 融合嵌入
```
POST /embeddings/fused
请求体: {"texts": ["文本1"], "images": ["url1"]}
返回: {"embeddings": [[...]], "dimension": 1024}
```

### 6. 相似度计算
```
POST /similarity
请求体: {"texts": ["文本1"], "images": ["url1"]}
返回: {"similarity_matrix": [[...]], "shape": [1, 1]}
```

## 预期性能

在 NVIDIA H800 (80GB) 上:

| 指标 | 原版本 | 批处理版本 |
|------|--------|-----------|
| 并发数 | 1 | 10-15 |
| QPS | ~2-5 | ~15-25 |
| 平均延迟 | ~200ms | ~300-500ms |
| P99 延迟 | ~300ms | ~800ms |
| 显存利用率 | ~25% | ~60% |

## 生产环境建议

1. **使用 systemd 管理服务**,确保自动重启
2. **配置日志轮转**,避免日志文件过大
3. **设置监控告警**,监控 GPU 利用率和 API 延迟
4. **定期备份模型文件**
5. **使用 Nginx 做反向代理**,添加限流和 HTTPS
6. **考虑使用容器化部署** (Docker/Kubernetes)

## 后续优化方向

1. 添加请求队列持久化
2. 实现多 GPU 支持
3. 添加模型热更新功能
4. 集成 Prometheus 监控
5. 添加请求缓存层
