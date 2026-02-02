"""
Base64 图片嵌入示例

演示如何使用 base64 编码的图片调用嵌入 API
"""

import base64
import requests
from pathlib import Path


def image_to_base64(image_path: str) -> str:
    """将图片文件转换为 base64 编码字符串"""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    return base64_str


def image_to_data_uri(image_path: str) -> str:
    """将图片文件转换为 data URI 格式"""
    # 根据文件扩展名确定 MIME 类型
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_types.get(ext, "image/png")

    base64_str = image_to_base64(image_path)
    return f"data:{mime_type};base64,{base64_str}"


def test_base64_embedding():
    """测试 base64 图片嵌入"""
    api_url = "http://localhost:8000"

    # 假设你有一张图片
    image_path = "test_image.jpg"

    # 方式 1: 纯 base64 字符串
    base64_str = image_to_base64(image_path)
    response = requests.post(
        f"{api_url}/embeddings/image",
        json={"images": [base64_str], "is_query": True},
    )
    print("方式 1 (纯 base64):")
    print(f"状态码: {response.status_code}")
    if response.ok:
        data = response.json()
        print(f"嵌入维度: {data['dimension']}")
        print(f"向量数量: {len(data['embeddings'])}")
    print()

    # 方式 2: data URI 格式
    data_uri = image_to_data_uri(image_path)
    response = requests.post(
        f"{api_url}/embeddings/image",
        json={"images": [data_uri], "is_query": True},
    )
    print("方式 2 (data URI):")
    print(f"状态码: {response.status_code}")
    if response.ok:
        data = response.json()
        print(f"嵌入维度: {data['dimension']}")
        print(f"向量数量: {len(data['embeddings'])}")
    print()

    # 方式 3: 混合使用（URL、本地路径、base64）
    response = requests.post(
        f"{api_url}/embeddings/image",
        json={
            "images": [
                "https://example.com/image.jpg",  # URL
                "/path/to/local/image.jpg",  # 本地路径
                base64_str,  # base64
            ],
            "is_query": True,
        },
    )
    print("方式 3 (混合使用):")
    print(f"状态码: {response.status_code}")
    if response.ok:
        data = response.json()
        print(f"嵌入维度: {data['dimension']}")
        print(f"向量数量: {len(data['embeddings'])}")


def test_fused_embedding_with_base64():
    """测试融合嵌入（文本 + base64 图片）"""
    api_url = "http://localhost:8000"
    image_path = "test_image.jpg"

    base64_str = image_to_base64(image_path)

    response = requests.post(
        f"{api_url}/embeddings/fused",
        json={
            "texts": ["这是一张图片"],
            "images": [base64_str],
        },
    )

    print("融合嵌入 (文本 + base64 图片):")
    print(f"状态码: {response.status_code}")
    if response.ok:
        data = response.json()
        print(f"嵌入维度: {data['dimension']}")
        print(f"向量数量: {len(data['embeddings'])}")


if __name__ == "__main__":
    print("=" * 60)
    print("Base64 图片嵌入测试")
    print("=" * 60)
    print()

    try:
        test_base64_embedding()
        print()
        test_fused_embedding_with_base64()
    except Exception as e:
        print(f"错误: {e}")
