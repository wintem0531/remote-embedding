import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from modelscope import AutoModel
from pydantic import BaseModel, Field
from transformers.utils.versions import require_version

# 禁用 tokenizers 的并行处理,避免多线程环境下的死锁警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 检查 transformers 版本
require_version(
    "transformers<4.52.0",
    "The remote code has some issues with transformers>=4.52.0, please downgrade: pip install transformers==4.51.3",
)


# 全局模型实例
gme = None


# Pydantic 模型定义
class TextEmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="要嵌入的文本列表")
    instruction: Optional[str] = Field(None, description="可选的指令文本")


class ImageEmbeddingRequest(BaseModel):
    images: List[str] = Field(..., description="图像 URL 或本地路径列表")
    is_query: bool = Field(True, description="是否为查询模式")


class FusedEmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="文本列表")
    images: List[str] = Field(..., description="图像 URL 或本地路径列表")


class SimilarityRequest(BaseModel):
    texts: Optional[List[str]] = Field(None, description="文本列表")
    images: Optional[List[str]] = Field(None, description="图像 URL 或本地路径列表")
    text_instruction: Optional[str] = Field(None, description="文本嵌入的指令")


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="嵌入向量列表")
    dimension: int = Field(..., description="嵌入向量维度")


class SimilarityResponse(BaseModel):
    similarity_matrix: List[List[float]] = Field(..., description="相似度矩阵")
    shape: List[int] = Field(..., description="矩阵形状 [行, 列]")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global gme

    # 启动时加载模型
    print("正在加载 GME 模型...")
    try:
        gme = AutoModel.from_pretrained(
            "iic/gme-Qwen2-VL-7B-Instruct", torch_dtype="float16", device_map="cuda", trust_remote_code=True
        )
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise

    yield

    # 关闭时清理
    print("正在清理资源...")
    gme = None


# 创建 FastAPI 应用
app = FastAPI(
    title="Remote Embedding API",
    description="基于 GME-Qwen2-VL-7B-Instruct 的文本和图像嵌入服务",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """健康检查端点"""
    return {"status": "ok", "message": "Remote Embedding API 正在运行", "model": "iic/gme-Qwen2-VL-7B-Instruct"}


@app.post("/embeddings/text", response_model=EmbeddingResponse)
async def get_text_embeddings(request: TextEmbeddingRequest):
    """获取文本嵌入向量"""
    if gme is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        embeddings = gme.get_text_embeddings(texts=request.texts, instruction=request.instruction)

        return EmbeddingResponse(embeddings=embeddings.tolist(), dimension=embeddings.shape[1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"嵌入生成失败: {str(e)}")


@app.post("/embeddings/image", response_model=EmbeddingResponse)
async def get_image_embeddings(request: ImageEmbeddingRequest):
    """获取图像嵌入向量"""
    if gme is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        embeddings = gme.get_image_embeddings(images=request.images, is_query=request.is_query)

        return EmbeddingResponse(embeddings=embeddings.tolist(), dimension=embeddings.shape[1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"嵌入生成失败: {str(e)}")


@app.post("/embeddings/fused", response_model=EmbeddingResponse)
async def get_fused_embeddings(request: FusedEmbeddingRequest):
    """获取融合嵌入向量"""
    if gme is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    if len(request.texts) != len(request.images):
        raise HTTPException(status_code=400, detail="文本和图像列表长度必须相同")

    try:
        embeddings = gme.get_fused_embeddings(texts=request.texts, images=request.images)

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
            text_embeddings = gme.get_text_embeddings(texts=request.texts, instruction=request.text_instruction)

        # 获取图像嵌入
        if request.images:
            image_embeddings = gme.get_image_embeddings(images=request.images, is_query=False)

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

    uvicorn.run(app, host="0.0.0.0", port=8000)
