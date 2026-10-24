# -*- coding: utf-8 -*-
"""
步骤 1：知识库结构化治理
=======================
生成 567 条原始答疑素材（含脏数据）并清洗沉淀为 519 条高质量知识库。

用法： python prepare_data.py
"""
import json

from src.data_pipeline import run

if __name__ == "__main__":
    report = run()
    print("=" * 60)
    print("知识库治理完成")
    print("=" * 60)
    print(json.dumps(report, ensure_ascii=False, indent=2))
