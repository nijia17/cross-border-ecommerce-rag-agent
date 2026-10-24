# -*- coding: utf-8 -*-
"""
步骤 2：文本切分 + 向量化 + 构建 FAISS 索引
==========================================
用法： python build_index.py
"""
from src.index_builder import build_faiss_index

if __name__ == "__main__":
    stats = build_faiss_index()
    print("=" * 60)
    print("FAISS 向量库构建完成")
    print("=" * 60)
    for k, v in stats.items():
        print(f"{k}: {v}")
