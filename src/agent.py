# -*- coding: utf-8 -*-
"""
ReAct 智能体（对应文档 5.3）
==========================
使用 LangChain 1.x 的 langchain_classic.create_react_agent + AgentExecutor，
由大模型自主在“知识库检索”与 5 类业务工具之间推理调用。
未配置大模型 API 时返回 None，由 qa_chain 的离线规则路由接管。
"""
from __future__ import annotations

import json
from functools import lru_cache

from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish

from config import FALLBACK_REPLY
from src.llm import get_llm
from src.prompts import REACT_PROMPT
from src.tools import BUSINESS_TOOLS


class JSONTolerantReActParser(ReActSingleInputOutputParser):
    """兼容多参数工具的 ReAct 解析器。

    经典 MRKL/ReAct 解析器把 Action Input 原样当字符串传给工具（只适配单入参工具）；
    而 DeepSeek 等模型面对多参数工具（汇率/翻译/税费）会输出 JSON 对象字符串。
    这里在不破坏单字符串入参的前提下，把形如 {"k": v} 的 Action Input 解析成 dict，
    使 @tool 的 pydantic 入参模型能正确接收结构化参数。
    """

    def parse(self, text: str) -> AgentAction | AgentFinish:
        result = super().parse(text)
        if isinstance(result, AgentAction) and isinstance(result.tool_input, str):
            candidate = result.tool_input.strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        result.tool_input = parsed
                except (json.JSONDecodeError, ValueError):
                    pass  # 解析失败保持原字符串，交由工具侧/handle_parsing_errors 兜底
        return result


@lru_cache(maxsize=1)
def get_agent_executor(verbose: bool = False):
    """构建 ReAct AgentExecutor；离线模式返回 None。"""
    llm = get_llm()
    if llm is None:
        return None

    # LangChain 1.x：经典 ReAct Agent 已迁移至 langchain_classic
    from langchain_classic.agents import AgentExecutor, create_react_agent

    prompt = REACT_PROMPT.partial(fallback=FALLBACK_REPLY)
    agent = create_react_agent(
        llm, BUSINESS_TOOLS, prompt,
        output_parser=JSONTolerantReActParser(),
    )
    return AgentExecutor(
        agent=agent,
        tools=BUSINESS_TOOLS,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=4,
        return_intermediate_steps=True,
    )
