"""
批处理性能测试脚本
用于验证批处理优化的效果
"""

import asyncio
import time
import httpx
import statistics


# 测试配置
API_BASE_URL = "http://localhost:8000"
CONCURRENT_REQUESTS = 20  # 并发请求数
TEST_TEXTS = [
    "这是一个测试句子",
    "人工智能正在改变世界",
    "深度学习模型训练需要大量数据",
    "FastAPI 是一个高性能的 Web 框架",
    "批处理可以显著提升吞吐量",
]


async def send_single_request(client: httpx.AsyncClient, request_id: int):
    """发送单个请求"""
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


async def test_concurrent_requests(num_requests: int):
    """测试并发请求"""
    print(f"\n{'=' * 70}")
    print(f"开始并发测试: {num_requests} 个请求")
    print(f"{'=' * 70}")

    async with httpx.AsyncClient() as client:
        # 创建并发任务
        tasks = [
            send_single_request(client, i) for i in range(num_requests)
        ]

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
        print(f"\n延迟统计:")
        print(f"  平均延迟: {statistics.mean(latencies):.3f} 秒")
        print(f"  中位延迟: {statistics.median(latencies):.3f} 秒")
        print(f"  最小延迟: {min(latencies):.3f} 秒")
        print(f"  最大延迟: {max(latencies):.3f} 秒")

        # 计算吞吐量
        total_embeddings = sum(r["num_embeddings"] for r in successful)
        throughput = total_embeddings / total_elapsed
        qps = len(successful) / total_elapsed

        print(f"\n吞吐量:")
        print(f"  嵌入向量/秒: {throughput:.2f}")
        print(f"  请求/秒 (QPS): {qps:.2f}")

    if failed:
        print(f"\n失败的请求:")
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

            results_summary.append({
                "concurrency": level,
                "success_rate": len(successful) / len(results),
                "avg_latency": avg_latency,
                "qps": qps,
            })

        # 等待一会儿再进行下一轮测试
        await asyncio.sleep(2)

    # 打印汇总
    print("\n\n" + "=" * 70)
    print("汇总报告")
    print("=" * 70)
    print(f"{'并发数':<10} {'成功率':<10} {'平均延迟(s)':<15} {'QPS':<10}")
    print("-" * 70)

    for r in results_summary:
        print(
            f"{r['concurrency']:<10} "
            f"{r['success_rate']:<10.2%} "
            f"{r['avg_latency']:<15.3f} "
            f"{r['qps']:<10.2f}"
        )


async def check_server_status():
    """检查服务器状态"""
    print("正在检查服务器状态...")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/", timeout=5.0)
            response.raise_for_status()
            data = response.json()

            print(f"✓ 服务器正在运行")
            print(f"  模型: {data.get('model')}")
            print(f"  批处理: {'启用' if data.get('batching_enabled') else '禁用'}")

            if data.get('batch_config'):
                config = data['batch_config']
                print(f"  批次大小: {config.get('max_batch_size')}")
                print(f"  等待时间: {config.get('max_wait_ms')} ms")

            # 获取统计信息
            stats_response = await client.get(f"{API_BASE_URL}/stats", timeout=5.0)
            if stats_response.status_code == 200:
                stats = stats_response.json()
                print(f"\n当前统计:")
                print(f"  总请求数: {stats['stats'].get('total_requests', 0)}")
                print(f"  批处理请求数: {stats['stats'].get('batched_requests', 0)}")
                print(f"  平均批次大小: {stats['stats'].get('avg_batch_size', 0):.2f}")

            return True

    except httpx.ConnectError:
        print(f"✗ 无法连接到服务器 ({API_BASE_URL})")
        print(f"  请先启动服务器: uvicorn main_batched:app --host 0.0.0.0 --port 8000")
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
    print("1. 快速测试 (20 个并发请求)")
    print("2. 全面测试 (多个并发级别)")
    print("3. 自定义并发数")

    try:
        choice = input("\n请输入选项 (1-3): ").strip()

        if choice == "1":
            await test_concurrent_requests(CONCURRENT_REQUESTS)
        elif choice == "2":
            await test_different_concurrency_levels()
        elif choice == "3":
            num = int(input("请输入并发数: ").strip())
            await test_concurrent_requests(num)
        else:
            print("无效的选项")

    except KeyboardInterrupt:
        print("\n\n测试已取消")
    except Exception as e:
        print(f"\n错误: {e}")

    print("\n测试完成!")


if __name__ == "__main__":
    asyncio.run(main())
