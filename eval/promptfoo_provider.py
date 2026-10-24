# -*- coding: utf-8 -*-
"""promptfoo Python Provider：把用例问题送入本项目 smart_answer 统一入口。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.qa_chain import smart_answer  # noqa: E402


def call_api(prompt, options=None, context=None):
    try:
        # 注意不要 strip 掉空输入用例：空串本身是一条边界断言
        return {"output": smart_answer(prompt or "")["answer"]}
    except Exception as exc:  # promptfoo 将异常输出标记为失败
        return {"error": str(exc)}
