"""
批处理性能测试脚本
用于验证批处理优化的效果
"""

import asyncio
import base64
import statistics
import time
from pathlib import Path

import httpx

# 测试配置
API_BASE_URL = "https://nb-sais.ai4s.com.cn:8443/ws-caab6dca-c3df-4264-bffb-fbaf8aa43c0a/project-19e5ada2-a7d0-4ed8-b803-07999cf05207/user-82c4697c-c0a4-4f67-9ada-89258482def5/vscode/0589d87d-14b5-494b-8e8a-c8e646fdc1d9/610515e5-f8d2-4ae6-b023-3c08a6f0ce77/proxy/8000"
# API_BASE_URL = "http://localhost:8000"
CONCURRENT_REQUESTS = 20  # 并发请求数
IMAGE_DIR = Path("/Users/songtao/Downloads/real_data/test/output/real_data/images")
TEST_TEXTS = [
    "这是一个测试句子",
    "人工智能正在改变世界",
    "深度学习模型训练需要大量数据",
    "FastAPI 是一个高性能的 Web 框架",
    "批处理可以显著提升吞吐量",
]


def load_test_images(num_images: int = 5):
    """从图片目录加载测试图片"""
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"图片目录不存在: {IMAGE_DIR}")

    image_files = list(IMAGE_DIR.glob("*.jpg"))[:num_images]

    if not image_files:
        raise ValueError(f"未找到图片文件: {IMAGE_DIR}")

    print(f"已加载 {len(image_files)} 张测试图片")
    return image_files


def image_to_base64(image_path: Path) -> str:
    """将图片转换为 base64 编码"""
    with open(image_path, "rb") as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode("utf-8")


async def send_single_request(client: httpx.AsyncClient, request_id: int):
    """发送单个文本请求"""
    start_time = time.time()

    try:
        response = await client.post(
            f"{API_BASE_URL}/embeddings/text",
            json={"texts": TEST_TEXTS[:2]},  # 每次发送 2 个文本
            timeout=30.0,
        )
        response.raise_for_status()

        elapsed = time.time() - start_time
        data = response.json()

        return {
            "request_id": request_id,
            "success": True,
            "elapsed": elapsed,
            "dimension": data.get("dimension"),
            "num_embeddings": len(data.get("embeddings", [])),
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "request_id": request_id,
            "success": False,
            "elapsed": elapsed,
            "error": str(e),
        }


async def send_single_image_request(
    client: httpx.AsyncClient, request_id: int, image_files: list[Path]
):
    """发送单个图片请求"""
    start_time = time.time()

    try:
        # 每次发送 2 张图片
        images_to_send = image_files[request_id % len(image_files) : (request_id % len(image_files)) + 2]
        if len(images_to_send) < 2:
            images_to_send = image_files[:2]

        # 转换为 base64
        images_base64 = [image_to_base64(img) for img in images_to_send]

        response = await client.post(
            f"{API_BASE_URL}/embeddings/image",
            json={"images": images_base64},
            timeout=30.0,
        )
        response.raise_for_status()

        elapsed = time.time() - start_time
        data = response.json()

        return {
            "request_id": request_id,
            "success": True,
            "elapsed": elapsed,
            "dimension": data.get("dimension"),
            "num_embeddings": len(data.get("embeddings", [])),
            "image_names": [img.name for img in images_to_send],
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "request_id": request_id,
            "success": False,
            "elapsed": elapsed,
            "error": str(e),
        }


async def test_concurrent_requests(num_requests: int):
    """测试并发请求"""
    print(f"\n{'=' * 70}")
    print(f"开始并发测试: {num_requests} 个请求")
    print(f"{'=' * 70}")

    async with httpx.AsyncClient() as client:
        # 创建并发任务
        tasks = [send_single_request(client, i) for i in range(num_requests)]

        # 记录开始时间
        start_time = time.time()

        # 并发执行
        results = await asyncio.gather(*tasks)

        # 记录结束时间
        total_elapsed = time.time() - start_time

    # 统计结果
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n{'=' * 70}")
    print("测试结果:")
    print(f"{'=' * 70}")
    print(f"总请求数: {num_requests}")
    print(f"成功: {len(successful)}")
    print(f"失败: {len(failed)}")
    print(f"总耗时: {total_elapsed:.2f} 秒")

    if successful:
        latencies = [r["elapsed"] for r in successful]
        print("\n延迟统计:")
        print(f"  平均延迟: {statistics.mean(latencies):.3f} 秒")
        print(f"  中位延迟: {statistics.median(latencies):.3f} 秒")
        print(f"  最小延迟: {min(latencies):.3f} 秒")
        print(f"  最大延迟: {max(latencies):.3f} 秒")

        # 计算吞吐量
        total_embeddings = sum(r["num_embeddings"] for r in successful)
        throughput = total_embeddings / total_elapsed
        qps = len(successful) / total_elapsed

        print("\n吞吐量:")
        print(f"  嵌入向量/秒: {throughput:.2f}")
        print(f"  请求/秒 (QPS): {qps:.2f}")

    if failed:
        print("\n失败的请求:")
        for r in failed[:5]:  # 只显示前 5 个失败
            print(f"  请求 #{r['request_id']}: {r['error']}")

    return results


async def test_concurrent_image_requests(num_requests: int, image_files: list[Path]):
    """测试并发图片请求"""
    print(f"\n{'=' * 70}")
    print(f"开始并发图片测试: {num_requests} 个请求")
    print(f"{'=' * 70}")

    async with httpx.AsyncClient() as client:
        # 创建并发任务
        tasks = [send_single_image_request(client, i, image_files) for i in range(num_requests)]

        # 记录开始时间
        start_time = time.time()

        # 并发执行
        results = await asyncio.gather(*tasks)

        # 记录结束时间
        total_elapsed = time.time() - start_time

    # 统计结果
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n{'=' * 70}")
    print("测试结果:")
    print(f"{'=' * 70}")
    print(f"总请求数: {num_requests}")
    print(f"成功: {len(successful)}")
    print(f"失败: {len(failed)}")
    print(f"总耗时: {total_elapsed:.2f} 秒")

    if successful:
        latencies = [r["elapsed"] for r in successful]
        print("\n延迟统计:")
        print(f"  平均延迟: {statistics.mean(latencies):.3f} 秒")
        print(f"  中位延迟: {statistics.median(latencies):.3f} 秒")
        print(f"  最小延迟: {min(latencies):.3f} 秒")
        print(f"  最大延迟: {max(latencies):.3f} 秒")

        # 计算吞吐量
        total_embeddings = sum(r["num_embeddings"] for r in successful)
        throughput = total_embeddings / total_elapsed
        qps = len(successful) / total_elapsed

        print("\n吞吐量:")
        print(f"  嵌入向量/秒: {throughput:.2f}")
        print(f"  请求/秒 (QPS): {qps:.2f}")

        # 显示处理的图片信息
        print("\n处理的图片样例:")
        for r in successful[:3]:
            print(f"  请求 #{r['request_id']}: {r['image_names']}")

    if failed:
        print("\n失败的请求:")
        for r in failed[:5]:  # 只显示前 5 个失败
            print(f"  请求 #{r['request_id']}: {r['error']}")

    return results


async def test_different_concurrency_levels():
    """测试不同并发级别"""
    print("\n" + "=" * 70)
    print("多级并发测试")
    print("=" * 70)

    concurrency_levels = [1, 5, 10, 15, 20]

    results_summary = []

    for level in concurrency_levels:
        print(f"\n\n测试并发级别: {level}")
        print("-" * 70)

        results = await test_concurrent_requests(level)
        successful = [r for r in results if r["success"]]

        if successful:
            avg_latency = statistics.mean([r["elapsed"] for r in successful])
            qps = len(successful) / max([r["elapsed"] for r in successful])

            results_summary.append(
                {
                    "concurrency": level,
                    "success_rate": len(successful) / len(results),
                    "avg_latency": avg_latency,
                    "qps": qps,
                }
            )

        # 等待一会儿再进行下一轮测试
        await asyncio.sleep(2)

    # 打印汇总
    print("\n\n" + "=" * 70)
    print("汇总报告")
    print("=" * 70)
    print(f"{'并发数':<10} {'成功率':<10} {'平均延迟(s)':<15} {'QPS':<10}")
    print("-" * 70)

    for r in results_summary:
        print(f"{r['concurrency']:<10} {r['success_rate']:<10.2%} {r['avg_latency']:<15.3f} {r['qps']:<10.2f}")


async def check_server_status():
    """检查服务器状态"""
    print("正在检查服务器状态...")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/", timeout=5.0)
            response.raise_for_status()
            data = response.json()

            print("✓ 服务器正在运行")
            print(f"  模型: {data.get('model')}")
            print(f"  批处理: {'启用' if data.get('batching_enabled') else '禁用'}")

            if data.get("batch_config"):
                config = data["batch_config"]
                print(f"  批次大小: {config.get('max_batch_size')}")
                print(f"  等待时间: {config.get('max_wait_ms')} ms")

            # 获取统计信息
            stats_response = await client.get(f"{API_BASE_URL}/stats", timeout=5.0)
            if stats_response.status_code == 200:
                stats = stats_response.json()
                print("\n当前统计:")
                print(f"  总请求数: {stats['stats'].get('total_requests', 0)}")
                print(f"  批处理请求数: {stats['stats'].get('batched_requests', 0)}")
                print(f"  平均批次大小: {stats['stats'].get('avg_batch_size', 0):.2f}")

            return True

    except httpx.ConnectError:
        print(f"✗ 无法连接到服务器 ({API_BASE_URL})")
        print("  请先启动服务器: uvicorn main_batched:app --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False


async def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("批处理性能测试")
    print("=" * 70)

    # 检查服务器
    if not await check_server_status():
        return

    print("\n请选择测试模式:")
    print("1. 文本 - 快速测试 (20 个并发请求)")
    print("2. 文本 - 全面测试 (多个并发级别)")
    print("3. 文本 - 自定义并发数")
    print("4. 图片 - 快速测试 (20 个并发请求)")
    print("5. 图片 - 自定义并发数")

    try:
        choice = input("\n请输入选项 (1-5): ").strip()

        if choice == "1":
            await test_concurrent_requests(CONCURRENT_REQUESTS)
        elif choice == "2":
            await test_different_concurrency_levels()
        elif choice == "3":
            num = int(input("请输入并发数: ").strip())
            await test_concurrent_requests(num)
        elif choice == "4":
            # 加载测试图片
            try:
                image_files = load_test_images(num_images=10)
                await test_concurrent_image_requests(CONCURRENT_REQUESTS, image_files)
            except Exception as e:
                print(f"图片测试失败: {e}")
        elif choice == "5":
            num = int(input("请输入并发数: ").strip())
            num_images = int(input("请输入加载的图片数量: ").strip())
            try:
                image_files = load_test_images(num_images=num_images)
                await test_concurrent_image_requests(num, image_files)
            except Exception as e:
                print(f"图片测试失败: {e}")
        else:
            print("无效的选项")

    except KeyboardInterrupt:
        print("\n\n测试已取消")
    except Exception as e:
        print(f"\n错误: {e}")

    print("\n测试完成!")


if __name__ == "__main__":
    asyncio.run(main())
