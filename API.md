# Remote Embedding API 使用文档

基于 GME-Qwen2-VL-7B-Instruct 的文本和图像嵌入服务。

## 启动服务

```bash
# 使用 uv 运行
uv run python main.py

# 或直接使用 uvicorn
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

服务将在 `http://localhost:8000` 启动。

访问 `http://localhost:8000/docs` 查看自动生成的 API 文档（Swagger UI）。

## API 端点

### 1. 健康检查

**GET** `/`

检查服务状态。

**响应示例:**
```json
{
  "status": "ok",
  "message": "Remote Embedding API 正在运行",
  "model": "iic/gme-Qwen2-VL-7B-Instruct"
}
```

### 2. 文本嵌入

**POST** `/embeddings/text`

生成文本的嵌入向量。

**请求体:**
```json
{
  "texts": [
    "The Tesla Cybertruck is a battery electric pickup truck.",
    "Alibaba office."
  ],
  "instruction": "Find an image that matches the given text."  // 可选
}
```

**响应示例:**
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "dimension": 3584
}
```

### 3. 图像嵌入

**POST** `/embeddings/image`

生成图像的嵌入向量。

**请求体:**
```json
{
  "images": [
    "https://example.com/image1.jpg",
    "/path/to/local/image2.jpg"
  ],
  "is_query": false  // 默认 true
}
```

**响应示例:**
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "dimension": 3584
}
```

### 4. 融合嵌入

**POST** `/embeddings/fused`

生成文本和图像的融合嵌入向量。

**请求体:**
```json
{
  "texts": [
    "The Tesla Cybertruck is a battery electric pickup truck.",
    "Alibaba office."
  ],
  "images": [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg"
  ]
}
```

**注意:** 文本和图像列表长度必须相同。

**响应示例:**
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "dimension": 3584
}
```

### 5. 相似度计算

**POST** `/similarity`

计算文本和图像之间的相似度矩阵。

**请求体:**
```json
{
  "texts": [
    "The Tesla Cybertruck is a battery electric pickup truck.",
    "Alibaba office."
  ],
  "images": [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg"
  ],
  "text_instruction": "Find an image that matches the given text."  // 可选
}
```

**响应示例:**
```json
{
  "similarity_matrix": [
    [0.328, 0.026],
    [0.094, 0.313]
  ],
  "shape": [2, 2]
}
```

## 使用示例

### Python 客户端

```python
import requests

BASE_URL = "http://localhost:8000"

# 文本嵌入
response = requests.post(
    f"{BASE_URL}/embeddings/text",
    json={"texts": ["Hello world"]}
)
embeddings = response.json()["embeddings"]

# 图像嵌入
response = requests.post(
    f"{BASE_URL}/embeddings/image",
    json={"images": ["https://example.com/image.jpg"]}
)

# 计算相似度
response = requests.post(
    f"{BASE_URL}/similarity",
    json={
        "texts": ["A car"],
        "images": ["https://example.com/car.jpg"]
    }
)
similarity = response.json()["similarity_matrix"]
```

### cURL

```bash
# 健康检查
curl http://localhost:8000/

# 文本嵌入
curl -X POST http://localhost:8000/embeddings/text \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world"],
    "instruction": "Represent the sentence"
  }'

# 图像嵌入
curl -X POST http://localhost:8000/embeddings/image \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["https://example.com/image.jpg"],
    "is_query": false
  }'
```

## 运行测试

```bash
# 确保服务正在运行
uv run python main.py

# 在另一个终端运行测试
uv run python test_api.py
```

## 错误处理

服务返回以下 HTTP 状态码：

- `200` - 请求成功
- `400` - 请求参数错误
- `500` - 服务器内部错误
- `503` - 模型未加载

错误响应示例：
```json
{
  "detail": "嵌入生成失败: ..."
}
```

## 性能优化建议

1. **批量处理**: 尽可能将多个文本/图像放在一个请求中
2. **GPU 配置**: 确保 CUDA 可用以提高推理速度
3. **模型缓存**: 模型在首次启动时加载，后续请求复用同一模型实例

## 系统要求

- CUDA 兼容的 GPU（推荐）
- 至少 16GB GPU 内存
- Python 3.12+
