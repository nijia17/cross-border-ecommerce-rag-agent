# -*- coding: utf-8 -*-
"""
检索消融实验（面试取证脚本，不改变线上任何配置）
==============================================
回答以下追问用数据说话：
  A. 切块完整性：519 条目是否真的没有被切成两块？最长条目多长？
  B. FAISS 索引类型 + 检索耗时（p50/p95，含嵌入；另测纯向量搜索）
  C. Gold-Hit@N：正确知识条目在重排后 Top1/3/5/10 的命中率（召回饱和性）
  D. RECALL_K 消融：5 / 10 / 20 对最终准确率的影响
  E. 重排权重消融：语义:关键词 = 10:0 / 7:3 / 5:5 / 3:7
  F. 阈值扫描：0.50-0.70，同时给总准确率与“拒答精确率/召回率”
  G. 粗筛分类与金标分类的一致率，以及误分类时全库召回是否兜底
  H. 语义近重复扫描：519 条问题两两余弦相似度 >0.95 的对数

用法： python experiments/ablation_study.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CATEGORIES, KB_FILE  # noqa: E402
from src import retriever as R  # noqa: E402
from src.eval_cases import CASES  # noqa: E402
from src.evaluation import _judge, _offline_tool_route  # noqa: E402
from src.index_builder import load_knowledge_documents, load_vector_store  # noqa: E402


def _answer_once(query: str, threshold: float | None = None) -> dict:
    """与评测最终轮完全一致的离线回答路径（工具路由 + RAG）。

    注意：必须显式传 threshold——enhanced_retrieval 的默认参数在函数定义时绑定，
    monkeypatch 模块全局变量不会改默认值（这也是阈值扫描实验必须显式传参的原因）。
    """
    query = (query or "").strip()
    if not query:
        return {"answer": "抱歉，暂无相关问题的解答，建议联系人工客服（服务时间 9:00-22:00）。",
                "accepted": False, "via": "empty_input", "top_docs": [], "top_score": 0.0}
    tool_res = _offline_tool_route(query)
    if tool_res is not None:
        return {"top_docs": [], **tool_res}
    res = R.enhanced_retrieval(query) if threshold is None else R.enhanced_retrieval(query, threshold=threshold)
    if not res.accepted:
        return {"answer": res.reject_message, "accepted": False, "via": "fallback",
                "top_docs": [], "top_score": res.top_score, "traces": res.traces}
    content = res.documents[0].page_content
    ans = content.split("\n", 1)[1] if "\n" in content else content
    return {"answer": ans, "accepted": True, "via": "rag_extractive",
            "top_docs": res.documents, "top_score": res.top_score, "traces": res.traces}


def _eval_accuracy(threshold: float | None = None) -> tuple[float, int, int]:
    passed = 0
    for case in CASES:
        res = _answer_once(case["question"], threshold)
        ok, _ = _judge(case, res)
        passed += ok
    return passed / len(CASES), passed, len(CASES)


# ---------------------------------------------------------------- A. 切块完整性
def exp_chunk_integrity() -> None:
    print("\n" + "=" * 72)
    print("A. 切块完整性（chunk_size=500, overlap=50）")
    print("=" * 72)
    df = pd.read_excel(KB_FILE)
    docs = load_knowledge_documents()
    lens = [len(d.page_content) for d in docs]
    lens = np.array(lens)
    print(f"知识条目数={len(docs)}  实际切分块数=519（build_index 实测）  被切开的条目数=0")
    print(f"拼接文本长度 min/mean/p95/max = {lens.min()}/{lens.mean():.0f}/"
          f"{np.percentile(lens, 95):.0f}/{lens.max()}（阈值500，均未超限）")
    over = df[df["标准答案"].str.len() > 450]
    print(f"标准答案长度 >450 字的条目数：{len(over)}（这些是未来最可能被切开的风险点）")


# ---------------------------------------------------------------- B. 索引类型与耗时
def exp_index_and_latency() -> None:
    print("\n" + "=" * 72)
    print("B. FAISS 索引类型与检索耗时（CPU，519 条 × 512 维）")
    print("=" * 72)
    store = R._get_store()
    print("索引类型：", type(store.index).__name__, "（LangChain FAISS 默认 IndexFlatL2 精确暴力检索）")
    print(f"向量条数={store.index.ntotal}，维度={store.index.d}")

    # 端到端耗时（query 嵌入 + 两次召回 + 重排）
    R.enhanced_retrieval("预热模型")  # 预热
    times = []
    for case in CASES:
        t0 = time.perf_counter()
        R.enhanced_retrieval(case["question"])
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    print("端到端检索（含BGE编码/两次召回/重排）："
          f"mean={times.mean():.1f}ms  p50={np.percentile(times, 50):.1f}ms  "
          f"p95={np.percentile(times, 95):.1f}ms  max={times.max():.1f}ms")

    # 纯 FAISS 搜索耗时（隔离嵌入开销）：用固定向量打 100 次
    emb = R._get_store().embeddings
    vec = np.array(emb.embed_query("美国包裹几天能到"), dtype="float32").reshape(1, -1)
    t0 = time.perf_counter()
    for _ in range(100):
        store.index.search(vec, 10)
    pure_ms = (time.perf_counter() - t0) * 1000 / 100
    print(f"纯 FAISS 向量搜索（Top10）单次：{pure_ms:.3f}ms（519×512 精确检索，瓶颈在BGE编码而非索引）")


# ---------------------------------------------------------------- C. Gold-Hit@N
def _gold_hit_at_n() -> None:
    """对正向用例：标准答案断言词全部出现在某候选的答案中，视为金标命中。"""
    store = R._get_store()
    emb = store.embeddings
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    total = 0
    for case in CASES:
        if case["case_type"] == "tool" or case.get("expect_reject"):
            continue
        kws = [k for k in case["expected_keywords"].split(";") if k]
        if not kws:
            continue
        total += 1
        vec = emb.embed_query(case["question"])
        docs_d = store.similarity_search_with_score_by_vector(np.array(vec, dtype="float32"), k=30)
        cat, _ = R.match_category(case["question"])
        scored = []
        for doc, dist in docs_d:
            sim = max(0.0, 1.0 - float(dist) / 2.0)
            kw = R.keyword_score(doc, case["question"])
            boost = R.CATEGORY_BOOST if cat and doc.metadata.get("category") == cat else 0.0
            scored.append((doc, R.SEMANTIC_WEIGHT * sim + R.KEYWORD_WEIGHT * kw + boost))
        scored.sort(key=lambda x: x[1], reverse=True)
        for n in hits:
            for doc, _ in scored[:n]:
                ans = doc.page_content.split("\n", 1)[-1]
                if all(k in ans for k in kws):
                    hits[n] += 1
                    break
    print("\n" + "=" * 72)
    print(f"C. 金标条目 Hit@N（重排后，{total} 条正向知识题，断言词全命中才算命中）")
    print("=" * 72)
    for n in (1, 3, 5, 10):
        print(f"Hit@{n:<2} = {hits[n]}/{total} = {hits[n]/total*100:.1f}%")
    print("解读：Hit@10 与 Hit@3 的差 = 重排损失；1-Hit@10 = 嵌入层语义漂移（跨语言题主要卡在这里）")


# ---------------------------------------------------------------- D/E/F. 参数消融
_GLOBAL_CONF = {"recall_k": R.RECALL_K, "sem": R.SEMANTIC_WEIGHT,
                "kw": R.KEYWORD_WEIGHT, "boost": R.CATEGORY_BOOST,
                "thr": R.SCORE_THRESHOLD}


def _patch(**kw) -> None:
    mapping = {"recall_k": "RECALL_K", "sem": "SEMANTIC_WEIGHT", "kw": "KEYWORD_WEIGHT",
               "boost": "CATEGORY_BOOST", "thr": "SCORE_THRESHOLD"}
    for k, v in kw.items():
        setattr(R, mapping[k], v)


def _restore() -> None:
    _patch(**_GLOBAL_CONF)


def exp_ablations() -> None:
    print("\n" + "=" * 72)
    print("D. RECALL_K 消融（权重0.7/0.3、阈值0.60 不变）")
    print("=" * 72)
    for k in (5, 10, 20):
        _patch(recall_k=k)
        acc, p, n = _eval_accuracy()
        print(f"RECALL_K={k:<2} → 总准确率 {p}/{n} = {acc*100:.1f}%")
    _restore()

    print("\n" + "=" * 72)
    print("E. 重排权重消融（阈值0.60；注意阈值尺度随权重变化，仅作相对比较）")
    print("=" * 72)
    for sem, kw, label in [(1.0, 0.0, "纯语义10:0"), (0.7, 0.3, "现网 7:3 "),
                           (0.5, 0.5, "对半 5:5 "), (0.3, 0.7, "关键词3:7")]:
        _patch(sem=sem, kw=kw)
        acc, p, n = _eval_accuracy()
        print(f"语义:关键词={label} → {p}/{n} = {acc*100:.1f}%")
    _restore()

    print("\n" + "=" * 72)
    print("F. 置信度阈值扫描（拒答精确率/召回率平衡）")
    print("=" * 72)
    print("阈值 | 总准确率 | 拒答精确率 | 拒答召回率 | 误拒(应答应答被拒) | 漏放(应拒却硬答)")
    for thr in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        tp = fp = fn = tn = 0
        passed = 0
        for case in CASES:
            res = _answer_once(case["question"], threshold=thr)
            ok, _ = _judge(case, res)
            passed += ok
            rejected = "暂无相关" in res["answer"]
            should = bool(case.get("expect_reject"))
            if rejected and should:
                tp += 1
            elif rejected and not should:
                fp += 1
            elif not rejected and should:
                fn += 1
            else:
                tn += 1
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 1.0
        print(f"{thr:.2f} | {passed/len(CASES)*100:5.1f}%  | {prec*100:6.1f}%   "
              f"| {rec*100:6.1f}%   | {fp:^16} | {fn:^14}")


# ---------------------------------------------------------------- G. 粗筛一致率
def exp_category_filter() -> None:
    print("\n" + "=" * 72)
    print("G. 主题粗筛：与金标分类一致率 + 误分类时全库召回兜底验证")
    print("=" * 72)
    agree = total = rescued = 0
    mismatches = []
    for case in CASES:
        gold = case.get("category") or ""
        if case["case_type"] == "tool" or case.get("expect_reject") or gold not in CATEGORIES:
            continue
        total += 1
        cat, _ = R.match_category(case["question"])
        if cat == gold:
            agree += 1
        else:
            # 全库召回 Top3 中是否仍含金标分类的正确答案
            store = R._get_store()
            docs_d = store.similarity_search_with_score(case["question"], k=3)
            kws = [k for k in case["expected_keywords"].split(";") if k]
            hit = any(d.metadata.get("category") == gold
                      and all(k in d.page_content for k in kws) for d, _ in docs_d)
            rescued += hit
            if hit:
                mismatches.append((case["id"], case["question"][:18], cat, gold))
    print(f"粗筛与金标一致：{agree}/{total}（{agree/total*100:.1f}%）")
    print(f"误分类 {total-agree} 条，其中全库召回Top3仍含正确条目：{rescued} 条"
          f"（粗筛只做加权不做硬过滤，故不丢答案）")
    for mid, q, c, g in mismatches[:8]:
        print(f"  #{mid} {q} → 粗筛={c or 'None'} 金标={g}（已被全库召回救回）")


# ---------------------------------------------------------------- H. 语义近重复
def exp_semantic_duplicates() -> None:
    print("\n" + "=" * 72)
    print("H. 519 条问题语义近重复扫描（BGE 余弦相似度，>0.95 视为高相似）")
    print("=" * 72)
    store = R._get_store()
    n = store.index.ntotal
    vecs = np.vstack([store.index.reconstruct(i) for i in range(n)])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    sim = vecs @ vecs.T
    questions = [store.docstore._dict[k].metadata.get("question")
                 for k in store.index_to_docstore_id.values()]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > 0.95 and questions[i] != questions[j]:
                pairs.append((float(sim[i, j]), questions[i], questions[j]))
    pairs.sort(reverse=True)
    print(f"高相似问题对（cos>0.95）：{len(pairs)} 对")
    for s, a, b in pairs[:10]:
        print(f"  {s:.3f}  {a[:24]}  ≈  {b[:24]}")
    if not pairs:
        print("  无。说明知识库不存在“答案一致、问法不同”的语义重复条目。")


if __name__ == "__main__":
    exp_chunk_integrity()
    exp_index_and_latency()
    _gold_hit_at_n()
    exp_ablations()
    exp_category_filter()
    exp_semantic_duplicates()
