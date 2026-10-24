# -*- coding: utf-8 -*-
"""
三段式增强检索（对应文档第四章）
=============================
用户问题
  ① 主题粗筛：业务分类关键词匹配，缩小向量检索范围；
  ② 语义召回：FAISS 向量检索 Top-10（L2 距离归一化为相似度）；
  ③ 关键词加权重排：中文二元词片（bigram）重合度 + 分类命中加权；
  ④ 置信度兜底：最高分低于阈值 → 返回统一拒答话术，抑制幻觉。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.documents import Document

from config import (
    CATEGORY_BOOST,
    CATEGORY_KEYWORDS,
    FINAL_TOP_K,
    KEYWORD_WEIGHT,
    RECALL_K,
    SCORE_THRESHOLD,
    SEMANTIC_WEIGHT,
)
from src.index_builder import load_vector_store

_CJK_RE = re.compile(r"[\u4e00-\u9fa5]+")
_ASCII_WORD_RE = re.compile(r"[a-z0-9]+")

# 口语化表达 → 知识库标准表达（第 2 轮迭代：补充业务同义词，提升模糊问题匹配）
_SYNONYM_EXPANSIONS: dict[str, list[str]] = {
    "快递": ["物流"], "几天": ["多久", "时效"], "到货": ["送达", "收到", "签收"],
    "多久到": ["时效", "送达"], "没收到": ["未收到", "丢件"], "没更新": ["不更新", "轨迹"],
    "付款": ["支付"], "付不了": ["拒绝", "失败"], "扣款": ["扣款", "请款"],
    "给钱": ["支付"], "交钱": ["支付", "缴税"],
    "咋整": ["怎么办", "什么原因"], "咋办": ["怎么办", "什么原因"], "怎么开": ["申请", "开具", "可以开"],
    "衣服": ["服装"], "鞋子": ["鞋"], "咋退": ["怎么退"],
    "不想要": ["拒收", "退货"], "退钱": ["退款"], "换一件": ["换货"],
    "交税": ["缴税", "税费"], "要交多少税": ["税费", "税率"],
    "券": ["优惠券"], "打折": ["折扣", "优惠"], "便宜点": ["差价", "价保"],
    "登录不上": ["无法登录", "锁定"], "封号": ["封禁"], "注销": ["注销"],
    "客服": ["客服"], "上班": ["服务时间"], "发票": ["发票", "收据", "凭证"],
    "违禁词": ["极限词", "违规"], "违规词": ["极限词", "违规"],
    "第一": ["极限词", "最高级", "绝对", "广告法"], "行不行": ["可以", "允许"],
    "咋": ["怎么"], "开发票": ["发票", "收据", "凭证", "开具", "获取"],
}


# ---------------------------------------------------------------- ① 主题粗筛
def match_category(query: str) -> tuple[str | None, int]:
    """按关键词命中数匹配业务分类，返回（分类, 命中数）；未命中返回（None, 0）。"""
    q = query.lower()
    best_cat, best_hits = None, 0
    for cat, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw.lower() in q)
        if hits > best_hits:
            best_cat, best_hits = cat, hits
    return best_cat, best_hits


# ---------------------------------------------------------------- ③ 关键词重排
def _term_set(text: str, expand: bool = False) -> set[str]:
    """无分词依赖的中文词片集合：中文连续串取二元字组（短串退化为单字）+ 英文词。

    expand=True 时按口语同义词表追加知识库标准表达词片（仅用于用户侧 query）。
    """
    text = text.lower()
    terms: set[str] = set()
    for run in _CJK_RE.findall(text):
        if len(run) == 1:
            terms.add(run)
        else:
            terms.update(run[i : i + 2] for i in range(len(run) - 1))
    terms.update(_ASCII_WORD_RE.findall(text))

    if expand:
        for colloquial, standards in _SYNONYM_EXPANSIONS.items():
            if colloquial in text:
                for std in standards:
                    for run in _CJK_RE.findall(std):
                        if len(run) == 1:
                            terms.add(run)
                        else:
                            terms.update(run[i : i + 2] for i in range(len(run) - 1))
    return terms


def keyword_score(doc: Document, query: str) -> float:
    """查询词片在文档中的覆盖率（取值 [0,1]）。

    以二元字组覆盖率为主（0.75），单字覆盖率为辅（0.25）做平滑回退，
    缓解极短口语化问题二元组稀疏导致的零分问题。
    """
    q_bi = _term_set(query, expand=True)
    if not q_bi:
        return 0.0
    d_bi = _term_set(doc.page_content)
    bi_cov = len(q_bi & d_bi) / len(q_bi)

    q_uni = {ch for ch in query if "\u4e00" <= ch <= "\u9fa5"}
    d_uni = {ch for ch in doc.page_content if "\u4e00" <= ch <= "\u9fa5"}
    uni_cov = len(q_uni & d_uni) / len(q_uni) if q_uni else 0.0
    return 0.75 * bi_cov + 0.25 * uni_cov


@dataclass
class RetrievalResult:
    accepted: bool
    documents: list[Document] = field(default_factory=list)
    reject_message: str | None = None
    category: str | None = None
    top_score: float = 0.0
    traces: list[dict] = field(default_factory=list)   # 各候选得分明细，便于评测/排错


@lru_cache(maxsize=1)
def _get_store():
    return load_vector_store()


def enhanced_retrieval(
    query: str,
    top_k: int = FINAL_TOP_K,
    threshold: float = SCORE_THRESHOLD,
) -> RetrievalResult:
    """执行三段式增强检索；低置信度时 accepted=False 并给出拒答话术。"""
    from config import FALLBACK_REPLY

    query = (query or "").strip()
    if not query:
        return RetrievalResult(accepted=False, reject_message=FALLBACK_REPLY)

    store = _get_store()

    # ① 主题粗筛
    category, cat_hits = match_category(query)

    # ② 语义召回：粗筛分类内召回 + 全库召回合并去重
    #    粗筛用于缩小范围并加权，但不因一次误分类而丢掉全库的强相关结果
    recalled: list[tuple[Document, float]] = []
    seen_keys: set[tuple] = set()

    def _take(results) -> None:
        for doc, dist in results:
            key = (doc.metadata.get("category"), doc.metadata.get("question"))
            if key not in seen_keys:
                seen_keys.add(key)
                recalled.append((doc, float(dist)))

    if category:
        try:
            _take(store.similarity_search_with_score(
                query, k=RECALL_K, filter={"category": category}
            ))
        except Exception:
            pass
    _take(store.similarity_search_with_score(query, k=RECALL_K))

    # ③ 关键词加权重排
    scored: list[tuple[Document, float, float, float]] = []
    for doc, l2_dist in recalled:
        # BGE 向量已归一化，L2 距离 ∈ [0,2]，换算为相似度 ∈ [0,1]
        sim = max(0.0, 1.0 - float(l2_dist) / 2.0)
        kw = keyword_score(doc, query)
        boost = CATEGORY_BOOST if category and doc.metadata.get("category") == category else 0.0
        final = SEMANTIC_WEIGHT * sim + KEYWORD_WEIGHT * kw + boost
        scored.append((doc, final, sim, kw))

    scored.sort(key=lambda x: x[1], reverse=True)

    traces = [
        {
            "category": doc.metadata.get("category"),
            "question": doc.metadata.get("question"),
            "final": round(final, 4),
            "semantic": round(sim, 4),
            "keyword": round(kw, 4),
        }
        for doc, final, sim, kw in scored[:top_k]
    ]
    top_score = scored[0][1] if scored else 0.0

    # ④ 置信度兜底
    if not scored or top_score < threshold:
        return RetrievalResult(
            accepted=False,
            reject_message=FALLBACK_REPLY,
            category=category,
            top_score=round(top_score, 4),
            traces=traces,
        )

    return RetrievalResult(
        accepted=True,
        documents=[doc for doc, _, _, _ in scored[:top_k]],
        category=category,
        top_score=round(top_score, 4),
        traces=traces,
    )
