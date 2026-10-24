# -*- coding: utf-8 -*-
"""
RAG 问答主链路（对应文档第六章）与统一问答门面
============================================
* generate_answer(query) —— 文档 6.2 规定的“增强检索 → 上下文拼接 → LLM 生成”；
  无大模型时降级为 top1 知识条目的抽取式回答（离线演示/检索评测可复现）。
* smart_answer(query)   —— 在线模式走 ReAct Agent（知识库 + 5 工具自主调用）；
  离线模式用规则识别工具意图并直接执行，确保无密钥也能演示全部能力。
"""
from __future__ import annotations

import re

from langchain_core.documents import Document

from config import FALLBACK_REPLY, LLM_AVAILABLE
from src.agent import get_agent_executor
from src.llm import get_llm
from src.prompts import RAG_PROMPT
from src.retriever import enhanced_retrieval
from src.tools import (
    content_compliance_check,
    exchange_rate_calculator,
    order_delivery_query,
    tax_estimator,
    translator,
)


def _doc_answer(doc: Document) -> str:
    """从“【分类】问题\n答案”格式的 chunk 中取答案正文。"""
    return doc.page_content.split("\n", 1)[1] if "\n" in doc.page_content else doc.page_content


# ============================================================ 文档 6.2：RAG 回答生成
def generate_answer(query: str) -> dict:
    """纯知识库 RAG 问答（不触发业务工具），返回结构化结果。"""
    result = enhanced_retrieval(query)
    if not result.accepted:
        return {"answer": result.reject_message or FALLBACK_REPLY,
                "accepted": False, "category": result.category,
                "top_score": result.top_score, "traces": result.traces, "via": "fallback"}

    context = "\n---\n".join(
        f"【{d.metadata.get('category')}】{d.metadata.get('question')}\n{_doc_answer(d)}"
        for d in result.documents
    )

    llm = get_llm()
    if llm is not None:
        messages = RAG_PROMPT.format_messages(context=context, question=query)
        answer = llm.invoke(messages).content.strip()
        via = "rag_llm"
    else:
        # 离线演示：直接返回置信度最高条目的标准答案
        answer = _doc_answer(result.documents[0])
        via = "rag_extractive"

    return {"answer": answer, "accepted": True, "category": result.category,
            "top_score": result.top_score, "traces": result.traces, "via": via,
            "documents": result.documents}


# ============================================================ 离线工具意图路由
_AMOUNT_RE = re.compile(r"(\d+(?:\.\d+)?)")
_FOREIGN = "(美元|欧元|英镑|日元|澳元|加元|港币|新币|USD|EUR|GBP|JPY|AUD|CAD|HKD|SGD)"
_QUOTE_RE = re.compile(r"[“\"'「『](.+?)[”\"'」』]")
_ORDER_ID_RE = re.compile(r"\b([A-Z]{1,4}\d{5,}[A-Z0-9]*|\d{10,20})\b")
_TAX_CAT_RE = re.compile(r"(服装|服饰|鞋|包|箱包|母婴|玩具|食品|保健品|数码|3c|手机|电脑|家电|化妆品|护肤品|彩妆|香水|酒水|手表|首饰)")


def _offline_tool_route(query: str) -> dict | None:
    """识别工具类意图并直接执行；非工具问题返回 None 交给知识库 RAG。"""
    q = query.strip()

    # ① 汇率换算
    m = re.search(rf"{_AMOUNT_RE.pattern}\s*{_FOREIGN}", q, flags=re.IGNORECASE)
    if m and re.search(r"(换成|兑换|等于|=|折合|是多少|多少)", q) and re.search(r"(人民币|cny|CNY|块)", q):
        return {"answer": exchange_rate_calculator.invoke(
                    {"amount": float(m.group(1)), "currency": m.group(2).upper(), "direction": "to_cny"}),
                "via": "tool:exchange_rate_calculator"}
    m_cny = re.search(r"(\d+(?:\.\d+)?)\s*(?:元|人民币|CNY|cny)", q)
    if m_cny and re.search(_FOREIGN, q, flags=re.IGNORECASE) and re.search(r"(换成|兑换|等于|折合|多少)", q):
        cur = re.search(_FOREIGN, q, flags=re.IGNORECASE).group(1)
        return {"answer": exchange_rate_calculator.invoke(
                    {"amount": float(m_cny.group(1)), "currency": cur.upper(), "direction": "from_cny"}),
                "via": "tool:exchange_rate_calculator"}

    # ② 文案合规检测
    if re.search(r"(合规|违禁|极限词|违规词|广告词)", q) or ("文案" in q and re.search(r"(检测|检查|看看|能不能|行不行|可以吗)", q)):
        quotes = _QUOTE_RE.findall(q)
        text = quotes[0] if quotes else q
        return {"answer": content_compliance_check.invoke({"text": text}),
                "via": "tool:content_compliance_check"}

    # ③ 多语种翻译
    if re.search(r"(翻译|译成|翻译成|translate)", q, flags=re.IGNORECASE):
        quotes = _QUOTE_RE.findall(q)
        text = quotes[0] if quotes else re.sub(r"(请|帮我|把|一下|翻译|译成|翻译成|translate|成英文|成日文|为英文|为日文|英文|日语|英语|日文)", "", q, flags=re.IGNORECASE).strip(" ，。:：")
        lang = "ja" if re.search(r"(日文|日语)", q) else "en"
        if not text:
            text = q
        return {"answer": translator.invoke({"text": text, "target_lang": lang}),
                "via": "tool:translator"}

    # ④ 订单时效查询（要求出现明确订单号）
    m_oid = _ORDER_ID_RE.search(q.upper())
    if m_oid and re.search(r"(订单|运单|物流|到哪|时效|几号|什么时候)", q):
        return {"answer": order_delivery_query.invoke({"order_id": m_oid.group(1)}),
                "via": "tool:order_delivery_query"}

    # ⑤ 税费估算：出现品类 + 金额 + 税相关词
    m_val = re.search(r"(\d+(?:\.\d+)?)\s*(?:元|块|RMB|CNY)", q, flags=re.IGNORECASE)
    m_cat = _TAX_CAT_RE.search(q)
    if m_val and m_cat and re.search(r"(税|税费|关税|到手)", q):
        return {"answer": tax_estimator.invoke(
                    {"category": m_cat.group(1), "value": float(m_val.group(1)), "country": "中国"}),
                "via": "tool:tax_estimator"}

    return None


# ============================================================ 统一问答门面
def smart_answer(query: str) -> dict:
    """系统统一入口：在线走 ReAct Agent，离线走规则路由 + 知识库抽取。"""
    query = (query or "").strip()
    if not query:
        return {"answer": FALLBACK_REPLY, "accepted": False, "via": "empty_input",
                "category": None, "top_score": 0.0, "traces": []}

    # 在线模式：ReAct Agent 自主决定“检索知识库 or 调用工具”
    if LLM_AVAILABLE:
        executor = get_agent_executor()
        try:
            output = executor.invoke({"input": query})
            return {"answer": output["output"], "accepted": True, "via": "react_agent",
                    "steps": [getattr(a.tool, "name", str(a.tool)) if hasattr(a, "tool") else str(a[0])
                              for a in output.get("intermediate_steps", [])]}
        except Exception as exc:
            # Agent 异常时兜底回纯 RAG
            res = generate_answer(query)
            res["agent_error"] = str(exc)
            return res

    # 离线模式：工具意图优先，命中工具直接执行；否则走知识库 RAG
    tool_result = _offline_tool_route(query)
    if tool_result is not None:
        return {"accepted": True, "category": None, "top_score": None,
                "traces": [], **tool_result}
    return generate_answer(query)
