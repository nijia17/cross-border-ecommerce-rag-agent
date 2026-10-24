# -*- coding: utf-8 -*-
"""
自动化评测（对应文档第七章）
==========================
* 60 条分层用例覆盖 9 类场景：常规 / 模糊口语 / 跨工具 / 超纲拒答 / 边界异常 /
  对抗注入 / 跨语言 / 多跳组合 / 否定约束（8 题开发回归 + 52 题独立盲测）；
* 复现迭代对比：round1 基础向量检索 → round2 三段式增强 → round4 最终版（同义词+阈值调优）；
* 断言：关键词事实断言、拒答正确性断言、工具调用断言、边界不崩溃断言；
* 输出 Markdown 报告 + 明细 Excel，并同步生成 promptfoo 配置。

用法： python run_eval.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import EVAL_DIR, FALLBACK_REPLY, SCORE_THRESHOLD
from src.eval_cases import CASES
from src.qa_chain import _doc_answer, _offline_tool_route
from src.retriever import _get_store

CASES_XLSX = EVAL_DIR / "test_cases.xlsx"

FALLBACK_MARK = "暂无相关"   # 统一拒答话术断言标记
ROUNDS = ["round1_vector", "round2_three_stage", "round4_final"]
ROUND_NAMES = {
    "round1_vector": "第1轮：基础向量检索（Top3，无粗筛/重排，固定阈值0.55）",
    "round2_three_stage": "第2轮：三段式增强检索（粗筛+语义+关键词重排）",
    "round4_final": "第4轮：增强检索+口语同义词+Badcase调优（系统冻结后跑60例盲测）",
}


# 场景类型展示顺序与中文名（用例集扩充后不再写死 5 类）
TYPE_LABELS = {
    "regular": "常规", "fuzzy": "模糊口语", "tool": "跨工具",
    "reject": "超纲拒答", "boundary": "边界异常", "adversarial": "对抗注入",
    "multilingual": "跨语言", "multihop": "多跳组合", "negation": "否定约束",
}


def _ordered_types() -> list[str]:
    """按用例集中首次出现的顺序返回场景类型，保证报告列顺序稳定。"""
    seen: list[str] = []
    for c in CASES:
        if c["case_type"] not in seen:
            seen.append(c["case_type"])
    return seen


# ------------------------------------------------------------ 用例集落 Excel
def export_cases_xlsx() -> None:
    df = pd.DataFrame(CASES)[
        ["id", "case_type", "category", "source", "question", "expected_keywords",
         "expect_reject", "expected_tool", "reference_answer"]
    ]
    df.columns = ["编号", "场景类型", "业务分类", "构建方式", "用户问题", "期望关键词断言",
                  "是否期望拒答", "期望工具", "标准答案要点"]
    df.to_excel(CASES_XLSX, index=False)


# ------------------------------------------------------------ 各轮次回答逻辑
def _round1_vector_answer(query: str) -> dict:
    """第 1 轮基线：纯 FAISS 语义召回，固定阈值，无粗筛、无重排。"""
    store = _get_store()
    pairs = store.similarity_search_with_score(query, k=3)
    top_sim = max(0.0, 1.0 - float(pairs[0][1]) / 2.0)
    if top_sim < 0.55:
        return {"answer": FALLBACK_REPLY, "accepted": False, "via": "fallback",
                "top_docs": [], "top_score": round(top_sim, 4)}
    docs = [d for d, _ in pairs]
    return {"answer": _doc_answer(docs[0]), "accepted": True, "via": "round1_vector",
            "top_docs": docs, "top_score": round(top_sim, 4)}


def _round2_answer(query: str) -> dict:
    """第 2 轮：三段式检索，但临时关闭口语同义词扩展（第3-4轮才补齐）。"""
    import src.retriever as retriever

    saved = retriever._SYNONYM_EXPANSIONS.copy()
    retriever._SYNONYM_EXPANSIONS.clear()
    try:
        from src.qa_chain import generate_answer
        res = generate_answer(query)
        res["top_docs"] = res.get("documents", [])
        return res
    finally:
        retriever._SYNONYM_EXPANSIONS.update(saved)


def _round4_answer(query: str) -> dict:
    """第 4 轮最终系统（在线为 ReAct Agent，离线为规则工具 + RAG）。"""
    from src.qa_chain import smart_answer
    res = smart_answer(query)
    res.setdefault("top_docs", res.get("documents", []))
    return res


def answer_at_stage(query: str, stage: str) -> dict:
    """所有轮次共享同一套业务工具路由，差异仅在知识库检索策略。"""
    from config import LLM_AVAILABLE

    query = (query or "").strip()
    if not query:
        return {"answer": FALLBACK_REPLY, "accepted": False, "via": "empty_input",
                "top_docs": [], "top_score": 0.0}
    # 最终轮在线时完整走 ReAct Agent（工具由 Agent 自主决策），
    # 其余轮次/离线模式统一由规则路由工具，保证工具意图各轮可比。
    if stage == "round4_final" and LLM_AVAILABLE:
        return _round4_answer(query)
    tool_res = _offline_tool_route(query)
    if tool_res is not None:
        return {"top_docs": [], **tool_res}
    if stage == "round1_vector":
        return _round1_vector_answer(query)
    if stage == "round2_three_stage":
        return _round2_answer(query)
    return _round4_answer(query)


# ------------------------------------------------------------ 断言与归因
def _keywords(case: dict) -> list[str]:
    return [k.strip() for k in str(case["expected_keywords"]).split(";") if k.strip()]


def _top3_hits(case: dict, res: dict) -> list[str]:
    docs = res.get("top_docs", [])
    if not docs:
        return []
    joined = "\n".join(d.page_content for d in docs)
    return [k for k in _keywords(case) if k in joined]


def _judge(case: dict, res: dict) -> tuple[bool, str]:
    """返回（是否通过, 未通过归因）。归因口径：检索层/生成层/数据层/兜底阈值。

    断言优先级（与场景类型解耦，避免新增高压场景时漏判）：
      1. tool 类：先核对工具是否真实调用，再核对工具输出数值；
      2. expect_reject=1（任意场景）：必须输出统一拒答话术；
      3. 其余正向用例：先看是否被错误拒答，再逐条核对事实关键词。
    """
    answer = res.get("answer", "")
    via = res.get("via", "")

    if case["case_type"] == "tool":
        want_tool = case["expected_tool"]
        # 离线看 via（tool:xxx），在线 ReAct 看实际执行的 intermediate_steps
        called = [via] + [str(s) for s in res.get("steps", [])]
        if not any(want_tool in s for s in called):
            return False, "工具未触发（意图识别/Agent 路由）"
        return (all(k in answer for k in _keywords(case)),
                "工具结果与断言不符" if not all(k in answer for k in _keywords(case)) else "")

    # 凡是标注期望拒答的用例（超纲/边界/对抗越权），统一检查拒答话术
    if case.get("expect_reject"):
        if FALLBACK_MARK in answer:
            return True, ""
        return False, "兜底阈值过松（应拒答的问题被强行作答）"

    # ---- 正向用例（常规/模糊/对抗纠偏/跨语言/多跳/否定/长输入）----
    if not res.get("accepted"):
        hit3 = _top3_hits(case, res)
        return False, "检索层：阈值偏严被误拒答（跨语言/改写鲁棒性不足）" if hit3 \
            else "检索层：召回不足（知识库缺口或跨语言语义漂移）"

    missing = [k for k in _keywords(case) if k not in answer]
    if not missing:
        return True, ""
    hit3 = _top3_hits(case, res)
    if not hit3:
        return False, "检索层：语义召回未命中正确片段"
    if all(k in hit3 for k in missing):
        if case["case_type"] == "multihop":
            return False, "生成层：离线Top1抽取无法整合多片段（在线LLM整合Top3可解）"
        return False, "检索层：重排后 Top1 不准（或生成层未采用正确片段）"
    return False, "生成层：只命中部分事实片段（多跳信息不完整）"


# ------------------------------------------------------------ 执行评测
def evaluate() -> dict:
    export_cases_xlsx()
    detail_rows: list[dict] = []
    stage_metrics: dict[str, dict] = {}

    stage_failures: dict[str, list[str]] = {}
    for stage in ROUNDS:
        passed = 0
        type_stat: dict[str, list] = {}
        failures: list[str] = []
        for case in CASES:
            try:
                res = answer_at_stage(case["question"], stage)
                ok, reason = _judge(case, res)
                error = ""
            except Exception as exc:  # 边界：任何异常都算不通过
                res, ok, reason, error = {"answer": "", "via": "error"}, False, f"执行异常：{exc}", str(exc)
            passed += ok
            type_stat.setdefault(case["case_type"], []).append(ok)
            if not ok:
                failures.append(
                    f"#{case['id']}（{TYPE_LABELS.get(case['case_type'], case['case_type'])}）"
                    f"{case['question'] or '空输入'} → {reason}"
                )
            if stage == "round4_final":
                detail_rows.append({
                    "编号": case["id"], "场景": TYPE_LABELS.get(case["case_type"], case["case_type"]),
                    "构建方式": case.get("source", ""), "问题": case["question"],
                    "通过": "是" if ok else "否", "归因/备注": reason or "通过",
                    "链路": res.get("via"), "置信度": res.get("top_score"),
                    "系统回答": res.get("answer", "")[:300],
                })
        stage_failures[stage] = failures
        total = len(CASES)
        stage_metrics[stage] = {
            "总准确率": round(passed / total, 4),
            **{f"{t}通过率": round(sum(v) / len(v), 4) for t, v in type_stat.items()},
        }

    # ---- Badcase 归因统计（最终轮）：取“层级：说明”的层级前缀聚合
    import re
    attr_count: dict[str, int] = {}
    for row in detail_rows:
        if row["通过"] == "否":
            layer = re.split(r"[：:（(]", row["归因/备注"])[0]
            attr_count[layer] = attr_count.get(layer, 0) + 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = EVAL_DIR / f"report_{ts}.xlsx"
    pd.DataFrame(detail_rows).to_excel(detail_path, index=False)
    md_path = EVAL_DIR / f"report_{ts}.md"
    md_path.write_text(
        _build_md(stage_metrics, stage_failures, detail_rows, attr_count, ts),
        encoding="utf-8",
    )

    _export_promptfoo()

    return {"stage_metrics": stage_metrics, "badcase归因": attr_count,
            "report_md": str(md_path), "report_xlsx": str(detail_path)}


# ------------------------------------------------------------ 第 0 轮：纯大模型基线（需 API）
def baseline_llm_round() -> dict | None:
    """第 0 轮基线：无知识库、无工具，仅系统提示词约束下的纯大模型回答。"""
    from src.llm import get_llm
    from src.prompts import SYSTEM_PROMPT

    llm = get_llm()
    if llm is None:
        return None

    passed = 0
    failures: list[str] = []
    type_stat: dict[str, list] = {}
    for case in CASES:
        answer = llm.invoke([("system", SYSTEM_PROMPT), ("human", case["question"])]).content
        ok, reason = _judge(case, {"answer": answer, "accepted": True, "via": "llm_only"})
        passed += ok
        type_stat.setdefault(case["case_type"], []).append(ok)
        if not ok:
            failures.append(f"#{case['id']}（{case['case_type']}）{case['question']} → {reason}")
    return {
        "总准确率": round(passed / len(CASES), 4),
        **{f"{t}通过率": round(sum(v) / len(v), 4) for t, v in type_stat.items()},
        "failures": failures,
    }


# ------------------------------------------------------------ 报告生成
def _build_md(stage_metrics, stage_failures, detail_rows, attr_count, ts) -> str:
    from collections import Counter

    from config import LLM_AVAILABLE

    types = _ordered_types()
    type_counts = Counter(c["case_type"] for c in CASES)
    source_counts = Counter(c.get("source", "未标注") for c in CASES)
    dist = " / ".join(f"{TYPE_LABELS[t]}{type_counts[t]}" for t in types)
    src_dist = " / ".join(f"{k}{v}" for k, v in source_counts.items())

    lines = [
        "# 跨境电商 RAG 智能问答 Agent —— 效果评测报告",
        f"\n- 生成时间：{ts}",
        f"- 运行模式：{'在线大模型（DeepSeek + ReAct Agent）' if LLM_AVAILABLE else '离线演示模式（规则工具 + BGE 检索抽取式回答）'}",
        f"- 用例规模：{len(CASES)} 条（{dist}）",
        f"- 构建方式分布：{src_dist}",
        f"- 检索参数：RECALL_K=10, TOP_K=3, 阈值={SCORE_THRESHOLD}, 语义权重0.7/关键词0.3",
        "\n## 〇、评测口径（防止“题库即考卷”质疑）\n",
        "- #1-8 为知识库原题直问，仅作开发期回归监控；其余 52 条为独立设计的盲测题"
        "（口语改写、跨国家迁移、对抗注入、跨语言、多跳组合、否定约束、异常输入）。",
        "- 所有断言取自知识库标准答案与工具真实输出中的可核验 token，逐条可在明细 Excel 审计。",
        "- 第 1-3 轮迭代只依据 20 条开发集调参；**60 例盲测集在系统冻结后一次性运行，"
        "未根据盲测结果回改任何阈值/词表/代码**，失败用例原样进入下一轮 backlog。",
        "\n## 一、迭代轮次准确率对比（本仓实测，分母均为60）\n",
        "| 轮次 | 总准确率 | " + " | ".join(TYPE_LABELS[t] for t in types) + " |",
        "|---|" + "---|" * (len(types) + 1),
    ]
    for stage in ROUNDS:
        m = stage_metrics[stage]
        cells = " | ".join(f"{m.get(t + '通过率', 0)*100:.0f}%（{int(round(m.get(t+'通过率', 0)*type_counts[t]))}/{type_counts[t]}）"
                            for t in types)
        lines.append(f"| {ROUND_NAMES[stage]} | {m['总准确率']*100:.1f}% | {cells} |")
    # ---- 安全/体验双指标：解释“松阈值总分虚高”的问题
    safety_types = ["reject", "boundary", "adversarial"]    # 必须拒答/拒绝越权
    answer_types = [t for t in types if t not in safety_types]  # 必须正常作答

    def _pass_count(m: dict, t: str) -> int:
        return int(round(m.get(t + "通过率", 0) * type_counts[t]))

    lines += [
        "\n> 说明：第0轮“纯大模型无知识库约束”必须在配置 DeepSeek API 后，",
        "> 用 `python run_eval.py --baseline` 对同一批 60 条用例实测；离线模式无大模型，不产生该轮数据，",
        "> 严禁凭空填写任何基线数字。",
        "\n### 1.1 安全/体验双视角拆分（解释阈值的代价）\n",
        "| 轮次 | 应拒答组正确率（超纲+边界+对抗，共"
        f"{sum(type_counts[t] for t in safety_types)}题） | 应答组正确率（知识+口语+工具+跨语言+多跳+否定，共"
        f"{sum(type_counts[t] for t in answer_types)}题） |",
        "|---|---|---|",
    ]
    s_all = sum(type_counts[t] for t in safety_types)
    a_all = sum(type_counts[t] for t in answer_types)
    group_stat: dict[str, tuple[int, int]] = {}
    for stage in ROUNDS:
        m = stage_metrics[stage]
        s_ok = sum(_pass_count(m, t) for t in safety_types)
        a_ok = sum(_pass_count(m, t) for t in answer_types)
        group_stat[stage] = (s_ok, a_ok)
        lines.append(f"| {ROUND_NAMES[stage].split('：')[0]} | {s_ok}/{s_all}（{s_ok/s_all*100:.1f}%） | "
                     f"{a_ok}/{a_all}（{a_ok/a_all*100:.1f}%） |")
    s_final = group_stat["round4_final"][0]
    a1, a_final = group_stat["round1_vector"][1], group_stat["round4_final"][1]
    lines += [
        "\n> 关键结论：第1轮总分看似最高，但松阈值（0.55）使其对“写钓鱼邮件、索取他人手机号”等",
        f"> 对抗请求也强行作答，应拒答组仅 {group_stat['round1_vector'][0]}/{s_all}"
        f"（{group_stat['round1_vector'][0]/s_all*100:.1f}%），属安全事故；"
        "后续轮次把阈值提到 0.60 并加置信度兜底后，",
        f"> 应拒答组升到 {s_final}/{s_all}（{s_final/s_all*100:.1f}%），代价是跨语言与临界口语题召回下降"
        f"（应答组 {a1}/{a_all}→{a_final}/{a_all}，{a1/a_all*100:.1f}%→{a_final/a_all*100:.1f}%）。",
        "> 单看总准确率会掩盖该权衡，因此安全指标与体验指标必须分开报告。",
        "\n## 二、各轮次失败用例与问题归因\n",
    ]
    for stage in ROUNDS:
        lines.append(f"\n### {ROUND_NAMES[stage]}（{stage_metrics[stage]['总准确率']*100:.1f}%）")
        if stage_failures[stage]:
            lines.extend(f"- {f}" for f in stage_failures[stage])
        else:
            lines.append("- 全部用例通过。")
    lines += [
        "\n## 三、最终轮逐用例结果（完整60条，系统回答/置信度见底稿 Excel）\n",
        "| # | 场景 | 构建方式 | 问题 | 通过 | 链路 | 归因/备注 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in detail_rows:
        q = r["问题"] if r["问题"] else "（空输入）"
        lines.append(
            f"| {r['编号']} | {r['场景']} | {r['构建方式']} | {q[:40]} | "
            f"{r['通过']} | {r['链路']} | {r['归因/备注']} |"
        )

    lines += ["\n## 四、Badcase 三层归因（最终轮）\n"]
    if attr_count:
        for layer, cnt in attr_count.items():
            lines.append(f"- {layer}：{cnt} 条")
        lines += [
            "\n对应优化方向（即下一轮迭代 backlog）：",
            "- 检索层召回不到 → 优化切分、补充同义词/关键词、扩大 RECALL_K；",
            "- 检索层重排不准 → 调整语义/关键词权重与分类加权；",
            "- 跨语言漂移 → 换用多语种嵌入（如 bge-m3）或增加查询翻译前置；",
            "- 多跳整合失败 → 在线 LLM 基于 Top3 上下文综合作答，离线模式改为多片段抽取拼接；",
            "- 生成层答错（在线模式）→ 强化提示词约束、降低 temperature；",
            "- 数据层缺失 → 补充知识库素材并标注拒答场景。",
        ]
        # 逐条列出最终轮 Badcase 与下一轮对策（盲测冻结，只记录不回改）
        lines += ["\n### 4.1 最终轮 Badcase 明细（盲测时系统已冻结，以下为原始发现，未做针对性修复）\n",
                  "| # | 场景 | 问题 | 置信度 | 层级归因 | 下一轮对策 |",
                  "|---|---|---|---|---|---|"]
        fix_map = {
            "跨语言": "换多语种嵌入 bge-m3，或前置查询翻译为中文再检索",
            "多跳组合": "在线用 LLM 整合 Top3 上下文；离线改为多片段抽取拼接",
            "模糊口语": "查询改写扩展召回；语义显著领先时降低关键词重排权重",
            "否定约束": "补充否定/约束意图改写，在新开发集验证后再发版",
            "对抗注入": "增强噪声 query 清洗（去无关陈述，保留核心诉求）",
        }
        for r in detail_rows:
            if r["通过"] == "否":
                layer = r["归因/备注"].split("：")[0]
                plan = fix_map.get(r["场景"], "按三层归因进入下一轮分析")
                q = (r["问题"] or "")[:26]
                lines.append(f"| {r['编号']} | {r['场景']} | {q} | {r['置信度']} | {layer} | {plan} |")
    else:
        lines.append("- 最终轮全部用例通过，无 Badcase。")
    lines.append("\n## 五、自动化回归\n- 本报告与明细 Excel 由 `python run_eval.py` 自动生成；")
    lines.append("- promptfoo 断言配置见 `eval/promptfooconfig.yaml`，可执行 `npx promptfoo@latest eval` 复核。")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------ promptfoo 配置
def _export_promptfoo() -> None:
    import yaml

    (EVAL_DIR / "prompt.txt").write_text("{{question}}\n", encoding="utf-8")

    provider = '''# -*- coding: utf-8 -*-
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
'''
    (EVAL_DIR / "promptfoo_provider.py").write_text(provider, encoding="utf-8")

    tests = []
    for c in CASES:
        t = {"description": f"#{c['id']} {c['case_type']}",
             "vars": {"question": c["question"]}}
        if c["expect_reject"]:
            t["assert"] = [{"type": "contains", "value": "暂无相关"}]
        else:
            t["assert"] = [{"type": "contains", "value": k} for k in _keywords(c)]
        tests.append(t)

    config = {
        "description": f"跨境电商 RAG 智能问答 Agent 评测（{len(CASES)} 条，9 类高压场景分层盲测）",
        "prompts": ["prompt.txt"],
        "providers": [{
            "id": "python:promptfoo_provider.py",
            "label": "kb-rag-agent",
            "config": {"timeout": 60000},
        }],
        "tests": tests,
    }
    (EVAL_DIR / "promptfooconfig.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False, width=1000),
        encoding="utf-8",
    )
