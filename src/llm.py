# -*- coding: utf-8 -*-
"""
大模型工厂（对应文档 2.2 / 5.3）
==============================
配置了 DEEPSEEK_API_KEY 时返回 DeepSeek（OpenAI 兼容协议）ChatModel；
未配置时返回 None，系统自动进入“离线演示模式”（工具规则计算 + 知识库抽取式回答），
保证无密钥环境下全链路依然可运行、可评测。
"""
from __future__ import annotations

from functools import lru_cache

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    LLM_AVAILABLE,
    LLM_MODEL,
    LLM_TEMPERATURE,
)


@lru_cache(maxsize=1)
def get_llm():
    """获取大模型单例；未配置密钥返回 None。"""
    if not LLM_AVAILABLE:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=LLM_TEMPERATURE,
        timeout=30,
        max_retries=2,
    )
