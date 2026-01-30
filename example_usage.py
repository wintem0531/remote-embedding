"""
批处理版本使用示例
演示如何调用批处理优化的 API
"""

import requests
import time


# API 配置
API_BASE_URL = "http://localhost:8000"


def example_text_embedding():
    """示例: 文本嵌入"""
    print("\n" + "=" * 60)
    print("示例 1: 文本嵌入")
    print("=" * 60)

    # 准备数据
    texts = [
        "人工智能正在改变世界",
        "深度学习是机器学习的一个分支",
        "FastAPI 是一个现代化的 Python Web 框架",
    ]

    # 发送请求
    start = time.time()
    response = requests.post(
        f"{API_BASE_URL}/embeddings/text",
        json={"texts": texts},
        timeout=30,
    )
    elapsed = time.time() - start

    # 打印结果
    if response.status_code == 200:
        data = response.json()
        print(f"✓ 成功获取嵌入向量")
        print(f"  文本数量: {len(texts)}")
        print(f"  向量维度: {data['dimension']}")
        print(f"  响应时间: {elapsed:.3f} 秒")
        print(f"  第一个向量前 5 维: {data['embeddings'][0][:5]}")
    else:
        print(f"✗ 请求失败: {response.status_code}")
        print(f"  错误信息: {response.text}")


def example_image_embedding():
    """示例: 图像嵌入"""
    print("\n" + "=" * 60)
    print("示例 2: 图像嵌入")
    print("=" * 60)

    # 准备图像 URL (示例)
    images = [
        "https://example.com/image1.jpg",  # 替换为实际图像 URL
        "https://example.com/image2.jpg",
    ]

    print(f"注意: 请将图像 URL 替换为实际可访问的地址")
    print(f"示例图像: {images}")

    # 发送请求
    try:
        start = time.time()
        response = requests.post(
            f"{API_BASE_URL}/embeddings/image",
            json={"images": images, "is_query": True},
            timeout=30,
        )
        elapsed = time.time() - start

        if response.status_code == 200:
            data = response.json()
            print(f"✓ 成功获取图像嵌入")
            print(f"  图像数量: {len(images)}")
            print(f"  向量维度: {data['dimension']}")
            print(f"  响应时间: {elapsed:.3f} 秒")
    except Exception as e:
        print(f"✗ 请求失败: {e}")


def example_similarity():
    """示例: 文本相似度计算"""
    print("\n" + "=" * 60)
    print("示例 3: 文本相似度计算")
    print("=" * 60)

    # 准备文本
    texts = [
        "机器学习是人工智能的核心技术",
        "深度学习使用神经网络进行学习",
        "今天天气很好",
    ]

    # 发送请求
    start = time.time()
    response = requests.post(
        f"{API_BASE_URL}/similarity",
        json={"texts": texts},
        timeout=30,
    )
    elapsed = time.time() - start

    # 打印结果
    if response.status_code == 200:
        data = response.json()
        print(f"✓ 成功计算相似度")
        print(f"  矩阵形状: {data['shape']}")
        print(f"  响应时间: {elapsed:.3f} 秒")
        print(f"\n相似度矩阵:")

        similarity = data['similarity_matrix']
        for i, row in enumerate(similarity):
            print(f"  文本 {i+1}: {[f'{v:.4f}' for v in row]}")

        print(f"\n分析:")
        print(f"  文本1 vs 文本2: {similarity[0][1]:.4f} (相关度较高)")
        print(f"  文本1 vs 文本3: {similarity[0][2]:.4f} (相关度较低)")
    else:
        print(f"✗ 请求失败: {response.status_code}")


def check_stats():
    """查看批处理统计信息"""
    print("\n" + "=" * 60)
    print("批处理统计信息")
    print("=" * 60)

    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)

        if response.status_code == 200:
            data = response.json()
            stats = data['stats']
            queues = data['queue_sizes']

            print(f"总请求数: {stats['total_requests']}")
            print(f"批处理请求数: {stats['batched_requests']}")
            print(f"平均批次大小: {stats['avg_batch_size']:.2f}")
            print(f"\n当前队列:")
            print(f"  文本队列: {queues['text']}")
            print(f"  图像队列: {queues['image']}")
            print(f"  融合队列: {queues['fused']}")
        else:
            print(f"无法获取统计信息")

    except Exception as e:
        print(f"错误: {e}")


def check_service():
    """检查服务状态"""
    print("\n" + "=" * 60)
    print("检查服务状态")
    print("=" * 60)

    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)

        if response.status_code == 200:
            data = response.json()
            print(f"✓ 服务正常运行")
            print(f"  模型: {data['model']}")
            print(f"  批处理: {'启用' if data['batching_enabled'] else '禁用'}")

            if data.get('batch_config'):
                config = data['batch_config']
                print(f"  批次大小: {config['max_batch_size']}")
                print(f"  等待时间: {config['max_wait_ms']} ms")

            return True
        else:
            print(f"✗ 服务响应异常: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"✗ 无法连接到服务 ({API_BASE_URL})")
        print(f"  请确保服务已启动:")
        print(f"  ./run_batched.sh")
        return False
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("批处理版本使用示例")
    print("=" * 60)

    # 检查服务
    if not check_service():
        return

    # 运行示例
    example_text_embedding()
    example_similarity()
    # example_image_embedding()  # 需要实际图像 URL

    # 查看统计
    check_stats()

    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
