import asyncio
import base64
import io
import os
import re
import tempfile
import time
import traceback
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from modelscope import AutoModel
from PIL import Image
from pydantic import BaseModel, Field
from transformers.utils.versions import require_version

# 禁用 tokenizers 的并行处理,避免多线程环境下的死锁警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 设置 PyTorch 内存分配器以减少内存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 检查 transformers 版本
require_version(
    "transformers<4.52.0",
    "The remote code has some issues with transformers>=4.52.0, please downgrade: pip install transformers==4.51.3",
)


# ==================== 配置参数 ====================
class BatchConfig:
    """批处理配置"""

    # 文本批次可以较大,因为文本嵌入占用内存较少
    TEXT_MAX_BATCH_SIZE = 16  # 文本最大批次大小
    # 图像批次需要较小,因为图像嵌入占用大量 GPU 内存
    IMAGE_MAX_BATCH_SIZE = 2  # 图像最大批次大小 (降低以避免 OOM)
    # 融合嵌入批次也需要较小
    FUSED_MAX_BATCH_SIZE = 2  # 融合嵌入最大批次大小
    MAX_WAIT_MS = 50  # 最大等待时间(毫秒)
    ENABLE_BATCHING = True  # 是否启用批处理


# ==================== 图片预处理工具 ====================
def process_image_input(image_input: str) -> str:
    """
    处理图片输入，支持 URL、本地路径和 base64 格式

    Args:
        image_input: 图片输入，可以是：
            - URL (http:// 或 https://)
            - 本地文件路径
            - Base64 编码 (data:image/...;base64,... 或纯 base64 字符串)

    Returns:
        可被模型处理的图片路径或 URL
    """
    try:
        # 如果是 URL，直接返回
        if image_input.startswith(("http://", "https://")):
            print(f"[DEBUG] 检测到图片 URL: {image_input[:50]}...")
            return image_input

        # 如果是本地文件路径且存在，直接返回
        # 注意:先检查长度,避免 base64 字符串导致 "File name too long" 错误
        if len(image_input) < 500:  # 合理的文件路径长度限制
            try:
                if Path(image_input).is_file():
                    print(f"[DEBUG] 检测到本地文件: {image_input}")
                    return image_input
            except OSError:
                # 路径无效,继续尝试作为 base64 处理
                pass

        # 尝试解析 base64 数据
        print(f"[DEBUG] 尝试解析 Base64 数据,长度: {len(image_input)}")

        # 处理 data URI 格式: data:image/png;base64,iVBORw0KG...
        if image_input.startswith("data:image"):
            # 提取 base64 部分
            match = re.match(r"data:image/[^;]+;base64,(.+)", image_input)
            if match:
                base64_data = match.group(1)
                print(f"[DEBUG] 从 data URI 提取 Base64 数据")
            else:
                raise ValueError("无效的 data URI 格式")
        else:
            # 假设是纯 base64 字符串
            base64_data = image_input
            print(f"[DEBUG] 使用纯 Base64 数据")

        # 解码 base64
        image_bytes = base64.b64decode(base64_data)
        print(f"[DEBUG] Base64 解码成功,图片大小: {len(image_bytes)} 字节")

        # 转换为 PIL Image 验证是否为有效图片
        image = Image.open(io.BytesIO(image_bytes))
        print(f"[DEBUG] PIL 图片加载成功: {image.format} {image.size}")

        # 保存为临时文件
        # 使用 delete=False 以便模型可以访问，程序结束时会自动清理
        suffix = f".{image.format.lower()}" if image.format else ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            image.save(tmp_file, format=image.format or "PNG")
            print(f"[DEBUG] 图片保存为临时文件: {tmp_file.name}")
            return tmp_file.name

    except Exception as e:
        print(f"[ERROR] 图片处理失败:")
        print(f"[ERROR] 错误类型: {type(e).__name__}")
        print(f"[ERROR] 错误信息: {str(e)}")
        traceback.print_exc()
        raise ValueError(f"无法处理图片输入: {str(e)}")


# ==================== Pydantic 模型定义 ====================
class TextEmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="要嵌入的文本列表")
    instruction: Optional[str] = Field(None, description="可选的指令文本")


class ImageEmbeddingRequest(BaseModel):
    images: List[str] = Field(
        ...,
        description="图像列表，支持三种格式：\n"
        "1. URL (http:// 或 https://)\n"
        "2. 本地文件路径\n"
        "3. Base64 编码 (data:image/...;base64,... 或纯 base64 字符串)",
    )
    is_query: bool = Field(True, description="是否为查询模式")


class FusedEmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="文本列表")
    images: List[str] = Field(
        ...,
        description="图像列表，支持 URL、本地路径或 base64 编码",
    )


class SimilarityRequest(BaseModel):
    texts: Optional[List[str]] = Field(None, description="文本列表")
    images: Optional[List[str]] = Field(
        None, description="图像列表，支持 URL、本地路径或 base64 编码"
    )
    text_instruction: Optional[str] = Field(None, description="文本嵌入的指令")


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="嵌入向量列表")
    dimension: int = Field(..., description="嵌入向量维度")


class SimilarityResponse(BaseModel):
    similarity_matrix: List[List[float]] = Field(..., description="相似度矩阵")
    shape: List[int] = Field(..., description="矩阵形状 [行, 列]")


# ==================== 批处理请求封装 ====================
class BatchRequest:
    """批处理请求封装"""

    def __init__(self, request_type: str, **kwargs):
        self.request_type = request_type  # 'text', 'image', 'fused'
        self.kwargs = kwargs
        self.future = asyncio.Future()
        self.timestamp = time.time()


# ==================== 批处理器 ====================
class BatchProcessor:
    """智能批处理器，支持多种嵌入类型"""

    def __init__(
        self,
        model,
        text_max_batch_size: int = 16,
        image_max_batch_size: int = 2,
        fused_max_batch_size: int = 2,
        max_wait_ms: int = 50,
    ):
        self.model = model
        self.text_max_batch_size = text_max_batch_size
        self.image_max_batch_size = image_max_batch_size
        self.fused_max_batch_size = fused_max_batch_size
        self.max_wait_seconds = max_wait_ms / 1000.0

        # 为不同类型的请求维护独立队列
        self.text_queue = deque()
        self.image_queue = deque()
        self.fused_queue = deque()

        # 处理状态标志
        self.text_processing = False
        self.image_processing = False
        self.fused_processing = False

        # 统计信息
        self.stats = {
            "total_requests": 0,
            "batched_requests": 0,
            "avg_batch_size": 0.0,
        }

    async def add_text_request(self, texts: List[str], instruction: Optional[str] = None) -> torch.Tensor:
        """添加文本嵌入请求"""
        request = BatchRequest("text", texts=texts, instruction=instruction)
        self.text_queue.append(request)
        self.stats["total_requests"] += 1

        if not self.text_processing:
            asyncio.create_task(self._process_text_batches())

        return await request.future

    async def add_image_request(self, images: List[str], is_query: bool = True) -> torch.Tensor:
        """添加图像嵌入请求"""
        request = BatchRequest("image", images=images, is_query=is_query)
        self.image_queue.append(request)
        self.stats["total_requests"] += 1

        if not self.image_processing:
            asyncio.create_task(self._process_image_batches())

        return await request.future

    async def add_fused_request(self, texts: List[str], images: List[str]) -> torch.Tensor:
        """添加融合嵌入请求"""
        request = BatchRequest("fused", texts=texts, images=images)
        self.fused_queue.append(request)
        self.stats["total_requests"] += 1

        if not self.fused_processing:
            asyncio.create_task(self._process_fused_batches())

        return await request.future

    async def _process_text_batches(self):
        """处理文本嵌入批次"""
        self.text_processing = True

        while self.text_queue:
            batch = await self._collect_batch(self.text_queue, self.text_max_batch_size)

            if batch:
                await self._execute_text_batch(batch)

            # 如果队列为空，等待一小段时间再检查
            if not self.text_queue:
                await asyncio.sleep(0.001)

        self.text_processing = False

    async def _process_image_batches(self):
        """处理图像嵌入批次"""
        self.image_processing = True

        while self.image_queue:
            batch = await self._collect_batch(self.image_queue, self.image_max_batch_size)

            if batch:
                await self._execute_image_batch(batch)

            if not self.image_queue:
                await asyncio.sleep(0.001)

        self.image_processing = False

    async def _process_fused_batches(self):
        """处理融合嵌入批次"""
        self.fused_processing = True

        while self.fused_queue:
            batch = await self._collect_batch(self.fused_queue, self.fused_max_batch_size)

            if batch:
                await self._execute_fused_batch(batch)

            if not self.fused_queue:
                await asyncio.sleep(0.001)

        self.fused_processing = False

    async def _collect_batch(self, queue: deque, max_batch_size: int) -> List[BatchRequest]:
        """从队列中收集批次"""
        batch = []
        deadline = time.time() + self.max_wait_seconds

        while len(batch) < max_batch_size and queue:
            # 如果已有请求且超时，立即处理
            if time.time() > deadline and batch:
                break

            batch.append(queue.popleft())

            # 如果队列空了，等待一小段时间看是否有新请求
            if not queue and len(batch) < max_batch_size:
                remaining_time = deadline - time.time()
                if remaining_time > 0:
                    await asyncio.sleep(min(0.001, remaining_time))
                else:
                    break

        return batch

    async def _execute_text_batch(self, batch: List[BatchRequest]):
        """执行文本嵌入批处理"""
        try:
            # 合并所有文本
            all_texts = []
            text_counts = []

            for req in batch:
                texts = req.kwargs["texts"]
                all_texts.extend(texts)
                text_counts.append(len(texts))

            print(f"[DEBUG] 开始处理文本批次: {len(all_texts)} 个文本")

            # 批量推理（在线程池中执行同步调用）
            instruction = batch[0].kwargs.get("instruction")  # 使用第一个请求的 instruction
            embeddings = await asyncio.to_thread(
                self.model.get_text_embeddings,
                texts=all_texts,
                instruction=instruction,
            )

            print(f"[DEBUG] 文本嵌入完成: {embeddings.shape}")

            # 分发结果
            offset = 0
            for req, count in zip(batch, text_counts):
                result = embeddings[offset : offset + count]
                req.future.set_result(result)
                offset += count

            # 更新统计
            self.stats["batched_requests"] += len(batch)
            self._update_avg_batch_size(len(batch))

        except Exception as e:
            print(f"[ERROR] 文本批处理执行失败:")
            print(f"[ERROR] 错误类型: {type(e).__name__}")
            print(f"[ERROR] 错误信息: {str(e)}")
            print(f"[ERROR] 错误堆栈:")
            traceback.print_exc()

            # 所有请求都失败
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

    async def _execute_image_batch(self, batch: List[BatchRequest]):
        """执行图像嵌入批处理"""
        try:
            # 合并所有图像
            all_images = []
            image_counts = []

            for req in batch:
                images = req.kwargs["images"]
                all_images.extend(images)
                image_counts.append(len(images))

            print(f"[DEBUG] 开始处理图像批次: {len(all_images)} 张图片")

            # 预处理图片输入（支持 URL、本地路径和 base64）
            processed_images = await asyncio.to_thread(
                lambda: [process_image_input(img) for img in all_images]
            )

            print(f"[DEBUG] 图片预处理完成,准备调用模型")

            # 清理 GPU 缓存以释放内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"[DEBUG] GPU 缓存已清理")

            # 批量推理
            is_query = batch[0].kwargs.get("is_query", True)
            embeddings = await asyncio.to_thread(
                self.model.get_image_embeddings,
                images=processed_images,
                is_query=is_query,
            )

            print(f"[DEBUG] 模型推理完成,嵌入向量形状: {embeddings.shape}")

            # 推理后再次清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 分发结果
            offset = 0
            for req, count in zip(batch, image_counts):
                result = embeddings[offset : offset + count]
                req.future.set_result(result)
                offset += count

            self.stats["batched_requests"] += len(batch)
            self._update_avg_batch_size(len(batch))

        except Exception as e:
            print(f"[ERROR] 图像批处理执行失败:")
            print(f"[ERROR] 错误类型: {type(e).__name__}")
            print(f"[ERROR] 错误信息: {str(e)}")
            print(f"[ERROR] 错误堆栈:")
            traceback.print_exc()

            # 清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

    async def _execute_fused_batch(self, batch: List[BatchRequest]):
        """执行融合嵌入批处理"""
        try:
            print(f"[DEBUG] 开始处理融合嵌入批次: {len(batch)} 个请求")

            # 清理 GPU 缓存以释放内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"[DEBUG] GPU 缓存已清理")

            # 融合嵌入通常需要配对处理，批处理较复杂
            # 这里采用简化策略：逐个处理但在线程池中并发
            tasks = []
            for req in batch:
                # 预处理图片输入
                images = req.kwargs["images"]
                processed_images = await asyncio.to_thread(
                    lambda imgs=images: [process_image_input(img) for img in imgs]
                )

                task = asyncio.to_thread(
                    self.model.get_fused_embeddings,
                    texts=req.kwargs["texts"],
                    images=processed_images,
                )
                tasks.append((req, task))

            # 等待所有任务完成
            for req, task in tasks:
                try:
                    result = await task
                    req.future.set_result(result)
                except Exception as e:
                    print(f"[ERROR] 融合嵌入任务失败: {str(e)}")
                    traceback.print_exc()
                    req.future.set_exception(e)

            # 推理后清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[DEBUG] 融合嵌入批次处理完成")
            self.stats["batched_requests"] += len(batch)

        except Exception as e:
            print(f"[ERROR] 融合嵌入批处理执行失败:")
            print(f"[ERROR] 错误类型: {type(e).__name__}")
            print(f"[ERROR] 错误信息: {str(e)}")
            print(f"[ERROR] 错误堆栈:")
            traceback.print_exc()

            # 清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

    def _update_avg_batch_size(self, batch_size: int):
        """更新平均批次大小"""
        current_avg = self.stats["avg_batch_size"]
        total_batches = self.stats["batched_requests"]

        if total_batches > 0:
            self.stats["avg_batch_size"] = (current_avg * (total_batches - batch_size) + batch_size) / total_batches

    def get_stats(self) -> dict:
        """获取统计信息"""
        return self.stats.copy()


# ==================== 全局实例 ====================
gme = None
batch_processor = None


# ==================== 生命周期管理 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global gme, batch_processor

    # 启动时加载模型
    print("=" * 60)
    print("正在加载 GME 模型...")
    print("=" * 60)

    try:
        # 您可以修改为本地路径，例如: "./models/gme-Qwen2-VL-7B-Instruct"
        model_path = "/inspire/ssd/project/sais-bio/public/Chinese_civilization/models/Qwen/gme-Qwen2-VL-7B-Instruct"
        # model_path = "iic/gme-Qwen2-VL-7B-Instruct"

        gme = AutoModel.from_pretrained(
            model_path,
            torch_dtype="float16",
            device_map="cuda",
            trust_remote_code=True,
        )

        print(f"✓ 模型加载成功: {model_path}")

        # 初始化批处理器
        if BatchConfig.ENABLE_BATCHING:
            batch_processor = BatchProcessor(
                model=gme,
                text_max_batch_size=BatchConfig.TEXT_MAX_BATCH_SIZE,
                image_max_batch_size=BatchConfig.IMAGE_MAX_BATCH_SIZE,
                fused_max_batch_size=BatchConfig.FUSED_MAX_BATCH_SIZE,
                max_wait_ms=BatchConfig.MAX_WAIT_MS,
            )
            print(f"✓ 批处理已启用:")
            print(f"  - 文本批次大小: {BatchConfig.TEXT_MAX_BATCH_SIZE}")
            print(f"  - 图像批次大小: {BatchConfig.IMAGE_MAX_BATCH_SIZE}")
            print(f"  - 融合批次大小: {BatchConfig.FUSED_MAX_BATCH_SIZE}")
            print(f"  - 等待时间: {BatchConfig.MAX_WAIT_MS}ms")
        else:
            print("✗ 批处理已禁用")

        print("=" * 60)

    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        raise

    yield

    # 关闭时清理
    print("正在清理资源...")
    gme = None
    batch_processor = None


# ==================== FastAPI 应用 ====================
app = FastAPI(
    title="Remote Embedding API (Batched)",
    description="基于 GME-Qwen2-VL-7B-Instruct 的文本和图像嵌入服务 (批处理优化版)",
    version="0.2.0",
    lifespan=lifespan,
)


# ==================== API 端点 ====================
@app.get("/")
async def root():
    """健康检查端点"""
    stats = batch_processor.get_stats() if batch_processor else {}
    return {
        "status": "ok",
        "message": "Remote Embedding API 正在运行 (批处理优化版)",
        "model": "iic/gme-Qwen2-VL-7B-Instruct",
        "batching_enabled": BatchConfig.ENABLE_BATCHING,
        "batch_config": {
            "text_max_batch_size": BatchConfig.TEXT_MAX_BATCH_SIZE,
            "image_max_batch_size": BatchConfig.IMAGE_MAX_BATCH_SIZE,
            "fused_max_batch_size": BatchConfig.FUSED_MAX_BATCH_SIZE,
            "max_wait_ms": BatchConfig.MAX_WAIT_MS,
        },
        "stats": stats,
    }


@app.get("/stats")
async def get_stats():
    """获取批处理统计信息"""
    if batch_processor is None:
        raise HTTPException(status_code=503, detail="批处理器未初始化")

    return {
        "stats": batch_processor.get_stats(),
        "queue_sizes": {
            "text": len(batch_processor.text_queue),
            "image": len(batch_processor.image_queue),
            "fused": len(batch_processor.fused_queue),
        },
    }


@app.post("/embeddings/text", response_model=EmbeddingResponse)
async def get_text_embeddings(request: TextEmbeddingRequest):
    """获取文本嵌入向量 (批处理优化)"""
    if gme is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        if BatchConfig.ENABLE_BATCHING and batch_processor:
            # 使用批处理
            embeddings = await batch_processor.add_text_request(texts=request.texts, instruction=request.instruction)
        else:
            # 直接调用
            embeddings = await asyncio.to_thread(
                gme.get_text_embeddings,
                texts=request.texts,
                instruction=request.instruction,
            )

        return EmbeddingResponse(embeddings=embeddings.tolist(), dimension=embeddings.shape[1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"嵌入生成失败: {str(e)}")


@app.post("/embeddings/image", response_model=EmbeddingResponse)
async def get_image_embeddings(request: ImageEmbeddingRequest):
    """获取图像嵌入向量 (批处理优化)"""
    if gme is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        print(f"[DEBUG] 收到图像嵌入请求: {len(request.images)} 张图片")

        if BatchConfig.ENABLE_BATCHING and batch_processor:
            print(f"[DEBUG] 使用批处理模式")
            embeddings = await batch_processor.add_image_request(images=request.images, is_query=request.is_query)
        else:
            print(f"[DEBUG] 使用直接调用模式")
            # 预处理图片输入
            processed_images = await asyncio.to_thread(
                lambda: [process_image_input(img) for img in request.images]
            )
            embeddings = await asyncio.to_thread(
                gme.get_image_embeddings,
                images=processed_images,
                is_query=request.is_query,
            )

        print(f"[DEBUG] 图像嵌入成功生成: {embeddings.shape}")
        return EmbeddingResponse(embeddings=embeddings.tolist(), dimension=embeddings.shape[1])
    except Exception as e:
        print(f"[ERROR] 图像嵌入生成失败:")
        print(f"[ERROR] 错误类型: {type(e).__name__}")
        print(f"[ERROR] 错误信息: {str(e)}")
        print(f"[ERROR] 错误堆栈:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"嵌入生成失败: {str(e)}")


@app.post("/embeddings/fused", response_model=EmbeddingResponse)
async def get_fused_embeddings(request: FusedEmbeddingRequest):
    """获取融合嵌入向量 (批处理优化)"""
    if gme is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    if len(request.texts) != len(request.images):
        raise HTTPException(status_code=400, detail="文本和图像列表长度必须相同")

    try:
        if BatchConfig.ENABLE_BATCHING and batch_processor:
            embeddings = await batch_processor.add_fused_request(texts=request.texts, images=request.images)
        else:
            # 预处理图片输入
            processed_images = await asyncio.to_thread(
                lambda: [process_image_input(img) for img in request.images]
            )
            embeddings = await asyncio.to_thread(
                gme.get_fused_embeddings, texts=request.texts, images=processed_images
            )

        return EmbeddingResponse(embeddings=embeddings.tolist(), dimension=embeddings.shape[1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"嵌入生成失败: {str(e)}")


@app.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity(request: SimilarityRequest):
    """计算文本-图像相似度矩阵"""
    if gme is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    if not request.texts and not request.images:
        raise HTTPException(status_code=400, detail="至少需要提供文本或图像")

    try:
        text_embeddings = None
        image_embeddings = None

        # 获取文本嵌入
        if request.texts:
            if BatchConfig.ENABLE_BATCHING and batch_processor:
                text_embeddings = await batch_processor.add_text_request(
                    texts=request.texts, instruction=request.text_instruction
                )
            else:
                text_embeddings = await asyncio.to_thread(
                    gme.get_text_embeddings,
                    texts=request.texts,
                    instruction=request.text_instruction,
                )

        # 获取图像嵌入
        if request.images:
            if BatchConfig.ENABLE_BATCHING and batch_processor:
                image_embeddings = await batch_processor.add_image_request(images=request.images, is_query=False)
            else:
                # 预处理图片输入
                processed_images = await asyncio.to_thread(
                    lambda: [process_image_input(img) for img in request.images]
                )
                image_embeddings = await asyncio.to_thread(
                    gme.get_image_embeddings, images=processed_images, is_query=False
                )

        # 计算相似度
        if text_embeddings is not None and image_embeddings is not None:
            similarity = (text_embeddings @ image_embeddings.T).tolist()
            shape = [len(request.texts or []), len(request.images or [])]
        elif text_embeddings is not None:
            similarity = (text_embeddings @ text_embeddings.T).tolist()
            shape = [len(request.texts or []), len(request.texts or [])]
        elif image_embeddings is not None:
            similarity = (image_embeddings @ image_embeddings.T).tolist()
            shape = [len(request.images or []), len(request.images or [])]
        else:
            raise ValueError("无法生成嵌入向量")

        return SimilarityResponse(similarity_matrix=similarity, shape=shape)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"相似度计算失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # 重要: 只使用 1 个 worker,避免重复加载模型
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # 必须为 1
        log_level="info",
    )
