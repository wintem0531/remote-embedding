"""API 测试脚本示例"""
import requests
import json


BASE_URL = "http://localhost:8000"


def test_health_check():
    """测试健康检查端点"""
    print("测试健康检查...")
    response = requests.get(f"{BASE_URL}/")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}\n")


def test_text_embeddings():
    """测试文本嵌入"""
    print("测试文本嵌入...")
    data = {
        "texts": [
            "The Tesla Cybertruck is a battery electric pickup truck built by Tesla, Inc. since 2023.",
            "Alibaba office."
        ]
    }
    response = requests.post(f"{BASE_URL}/embeddings/text", json=data)
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"嵌入维度: {result['dimension']}")
    print(f"嵌入数量: {len(result['embeddings'])}\n")


def test_image_embeddings():
    """测试图像嵌入"""
    print("测试图像嵌入...")
    data = {
        "images": [
            "https://upload.wikimedia.org/wikipedia/commons/e/e9/Tesla_Cybertruck_damaged_window.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/e/e0/TaobaoCity_Alibaba_Xixi_Park.jpg"
        ],
        "is_query": False
    }
    response = requests.post(f"{BASE_URL}/embeddings/image", json=data)
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"嵌入维度: {result['dimension']}")
    print(f"嵌入数量: {len(result['embeddings'])}\n")


def test_similarity():
    """测试相似度计算"""
    print("测试文本-图像相似度...")
    data = {
        "texts": [
            "The Tesla Cybertruck is a battery electric pickup truck built by Tesla, Inc. since 2023.",
            "Alibaba office."
        ],
        "images": [
            "https://upload.wikimedia.org/wikipedia/commons/e/e9/Tesla_Cybertruck_damaged_window.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/e/e0/TaobaoCity_Alibaba_Xixi_Park.jpg"
        ],
        "text_instruction": "Find an image that matches the given text."
    }
    response = requests.post(f"{BASE_URL}/similarity", json=data)
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"相似度矩阵形状: {result['shape']}")
    print(f"相似度矩阵:")
    for row in result['similarity_matrix']:
        print(f"  {row}\n")


def test_fused_embeddings():
    """测试融合嵌入"""
    print("测试融合嵌入...")
    data = {
        "texts": [
            "The Tesla Cybertruck is a battery electric pickup truck built by Tesla, Inc. since 2023.",
            "Alibaba office."
        ],
        "images": [
            "https://upload.wikimedia.org/wikipedia/commons/e/e9/Tesla_Cybertruck_damaged_window.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/e/e0/TaobaoCity_Alibaba_Xixi_Park.jpg"
        ]
    }
    response = requests.post(f"{BASE_URL}/embeddings/fused", json=data)
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"嵌入维度: {result['dimension']}")
    print(f"嵌入数量: {len(result['embeddings'])}\n")


if __name__ == "__main__":
    try:
        test_health_check()
        test_text_embeddings()
        test_image_embeddings()
        test_similarity()
        test_fused_embeddings()
        print("所有测试完成！")
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到 API 服务。请确保服务正在运行：")
        print("  uv run python main.py")
    except Exception as e:
        print(f"测试失败: {e}")
