# -*- coding: utf-8 -*-
"""
知识库文本切分、向量化与 FAISS 索引构建（对应文档 3.3）
=====================================================
读取 data/knowledge_base.xlsx（519 条）→ 拼接检索文本 →
递归字符切分（chunk=500, overlap=50）→ BGE 向量化 → FAISS 持久化。
"""
from __future__ import annotations

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    FAISS_DIR,
    KB_FILE,
    SPLIT_SEPARATORS,
)
from src.embeddings import get_embeddings


def load_knowledge_documents() -> list[Document]:
    """读取清洗后的知识库 Excel，封装为 LangChain Document（含元数据）。"""
    if not KB_FILE.exists():
        raise FileNotFoundError(f"未找到知识库 {KB_FILE}，请先运行 python prepare_data.py")

    df = pd.read_excel(KB_FILE)
    docs: list[Document] = []
    for _, row in df.iterrows():
        category = str(row["业务分类"])
        question = str(row["问题"]).strip()
        answer = str(row["标准答案"]).strip()
        # 检索文本：分类 + 问题 + 答案（与文档口径一致）
        content = f"【{category}】{question}\n{answer}"
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "category": category,
                    "question": question,
                    "region": str(row.get("适用地区", "通用")),
                    "updated": str(row.get("更新时间", "")),
                },
            )
        )
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """递归字符切分，按 换行→句号→分号 等分隔符递归，保证语义完整。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SPLIT_SEPARATORS,
        keep_separator=True,
    )
    # split_documents 会保留每个来源 Document 的元数据
    return splitter.split_documents(docs)


def build_faiss_index() -> dict:
    """构建并持久化 FAISS 向量库，返回构建统计信息。"""
    docs = load_knowledge_documents()
    chunks = split_documents(docs)

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(FAISS_DIR))

    stats = {
        "知识条目数": len(docs),
        "切分后块数": len(chunks),
        "索引目录": str(FAISS_DIR),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    return stats


def load_vector_store() -> FAISS:
    """加载已构建的 FAISS 向量库。"""
    if not (FAISS_DIR / "index.faiss").exists():
        raise FileNotFoundError("FAISS 索引不存在，请先运行 python build_index.py")
    return FAISS.load_local(
        str(FAISS_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,   # 加载本地自构建索引
    )


if __name__ == "__main__":
    print(build_faiss_index())
