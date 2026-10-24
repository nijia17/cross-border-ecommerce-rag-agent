# -*- coding: utf-8 -*-
"""
嵌入模型工厂（对应文档 3.3 / 技术栈表）
=====================================
优先级：
1. 配置了 DASHSCOPE_API_KEY  → 阿里云百炼在线 Embedding（可选）；
2. 默认使用本地 BAAI/bge-small-zh-v1.5（中文、512 维、CPU 即可，完全离线）；
3. 本地缓存中仅有 bge-small-zh（v1）时自动降级，保证离线可运行。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from config import BGE_MODEL_NAME, DASHSCOPE_API_KEY, EMBED_DEVICE


def _hf_cache_has(model_repo: str) -> bool:
    """检查 HuggingFace 本地缓存中是否存在完整模型快照。"""
    hub = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    model_dir = hub / f"models--{model_repo.replace('/', '--')}"
    if not model_dir.exists():
        return False
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        return False
    for snap in snapshots.iterdir():
        # 至少存在配置文件与任一权重文件，即视为可离线加载
        if (snap / "config.json").exists() and any(
            (snap / w).exists() for w in ("model.safetensors", "pytorch_model.bin", "onnx")
        ):
            return True
    return False


def _resolve_bge_model() -> str:
    """决定最终加载的 BGE 模型：显式配置 > v1.5 缓存 > v1 缓存 > 默认仓库名。"""
    if os.getenv("BGE_MODEL"):
        return BGE_MODEL_NAME
    if _hf_cache_has("BAAI/bge-small-zh-v1.5"):
        return "BAAI/bge-small-zh-v1.5"
    if _hf_cache_has("BAAI/bge-small-zh"):
        return "BAAI/bge-small-zh"
    return BGE_MODEL_NAME


@lru_cache(maxsize=1)
def get_embeddings():
    """获取嵌入模型单例。"""
    # 可选：阿里云百炼在线 Embedding
    if DASHSCOPE_API_KEY:
        try:
            from langchain_community.embeddings import DashScopeEmbeddings

            return DashScopeEmbeddings(model="text-embedding-v3")
        except Exception as exc:  # 未安装 dashscope 时降级到本地 BGE
            print(f"[embeddings] DashScope 不可用，降级本地 BGE：{exc}")

    # 默认：本地 BGE（首次使用若未缓存会自动下载，可设置 HF_ENDPOINT 镜像）
    model_name = _resolve_bge_model()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True},   # 归一化后向量便于余弦相似度
    )
    print(f"[embeddings] 已加载嵌入模型：{model_name}（device={EMBED_DEVICE}）")
    return embeddings
