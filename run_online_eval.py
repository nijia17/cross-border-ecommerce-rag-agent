# -*- coding: utf-8 -*-
"""
在线大模型实测脚本（需配置 .env 的 DEEPSEEK_API_KEY）
==================================================
补齐离线模式无法验证的三件事（对应面试体检第三、五章）：
  ① 第0轮基线：纯大模型 + 系统提示词约束，【无知识库、无工具】（60 条）
  ② 第4轮在线：完整 ReAct Agent（知识库三段式检索 + 5 工具自主调用，60 条）
  ③ temperature 消融：0 / 0.1 / 0.3 在 12 条代表题上的关键词准确率与篇幅

所有模型回答原样落 Excel，失败用例打印实际回答，供人工区分
“真实错误”与“仅措辞不同但语义正确”（自动关键词断言的已知盲区）。

用法： python run_online_eval.py
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
from langchain_openai import ChatOpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    EVAL_DIR,
    FALLBACK_REPLY,
    LLM_MODEL,
)
from src.eval_cases import CASES
from src.evaluation import TYPE_LABELS, _judge, _keywords
from src.prompts import SYSTEM_PROMPT
from src.qa_chain import smart_answer
from src.retriever import enhanced_retrieval

FALLBACK_MARK = "暂无相关"


# ================================================================ ① 第0轮基线
def run_round0_baseline(max_workers: int = 8) -> list[dict]:
    """纯 LLM：只有系统提示词，不提供任何知识库片段与工具。"""
    llm = ChatOpenAI(model=LLM_MODEL, api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL,
                     temperature=0.1, timeout=60, max_retries=2)

    def _one(case: dict) -> dict:
        q = case["question"]
        if not q.strip():
            answer = FALLBACK_REPLY  # 空输入由系统输入守卫直接兜底，与模型无关
            via = "empty_guard"
        else:
            t0 = time.time()
            msg = llm.invoke([("system", SYSTEM_PROMPT), ("human", q)])
            answer = msg.content.strip()
            via = f"llm_only ({time.time()-t0:.0f}s)"
        ok, reason = _judge(case, {"answer": answer, "accepted": True, "via": "llm_only"})
        return {"id": case["id"], "type": case["case_type"], "question": q,
                "answer": answer, "ok": ok, "reason": reason, "via": via}

    return _run_pool(_one, max_workers, name="第0轮纯LLM基线")


# ================================================================ ② 第4轮在线
def run_round4_online(max_workers: int = 4) -> list[dict]:
    """完整在线系统：ReAct Agent 自主检索/调工具/拒答。"""

    def _one(case: dict) -> dict:
        q = case["question"]
        t0 = time.time()
        try:
            res = smart_answer(q)
            secs = time.time() - t0
            answer = res.get("answer", "")
            steps = [str(s) for s in res.get("steps", [])]
            via = res.get("via", "")
            err = str(res.get("agent_error", ""))[:150]
        except Exception as exc:  # 单条异常不拖垮整批评测
            answer, steps, via, err, secs = "", [], "exception", f"{type(exc).__name__}: {exc}"[:150], time.time() - t0
        ok, reason = _judge(case, {"answer": answer, "accepted": True,
                                   "via": via, "steps": steps})
        return {"id": case["id"], "type": case["case_type"], "question": q,
                "answer": answer, "ok": ok, "reason": reason, "via": via,
                "steps": " | ".join(s.split("tool=")[-1].split(" ")[0].strip("'\"")
                                    for s in steps if "tool=" in s),
                "secs": round(secs, 1), "err": err}

    return _run_pool(_one, max_workers, name="第4轮在线Agent")


def _run_pool(fn, max_workers: int, name: str) -> list[dict]:
    rows: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(fn, c): c["id"] for c in CASES}
        done = 0
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if done % 10 == 0:
                print(f"  [{name}] {done}/{len(CASES)} ...")
    rows.sort(key=lambda x: x["id"])
    print(f"[{name}] 完成，用时 {(time.time()-t0)/60:.1f} 分钟")
    return rows


# ================================================================ ③ temperature 消融
TEMP_QUESTIONS = [1, 2, 5, 9, 13, 17, 19, 25, 49, 56, 57, 60]


def run_temperature_ablation() -> pd.DataFrame:
    """同一份检索上下文，仅改 temperature，对比关键词准确率与篇幅。"""
    from langchain_core.prompts import ChatPromptTemplate

    from src.prompts import RAG_PROMPT

    cases = [c for c in CASES if c["id"] in TEMP_QUESTIONS]
    # 所有温度共享完全相同的检索上下文（只检索一次）
    prepared = []
    for c in cases:
        r = enhanced_retrieval(c["question"])
        if not r.accepted:
            continue
        context = "\n---\n".join(
            f"【{d.metadata.get('category')}】{d.metadata.get('question')}\n"
            f"{d.page_content.split(chr(10), 1)[-1]}" for d in r.documents
        )
        prepared.append((c, context))

    records = []
    for temp in (0.0, 0.1, 0.3):
        llm = ChatOpenAI(model=LLM_MODEL, api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL,
                         temperature=temp, timeout=60, max_retries=2)

        def _one(item, temp=temp):
            c, context = item
            msgs = RAG_PROMPT.format_messages(context=context, question=c["question"])
            ans = llm.invoke(msgs).content.strip()
            kws = _keywords(c)
            hit = sum(k in ans for k in kws) / len(kws) if kws else 1.0
            return c["id"], TYPE_LABELS.get(c["case_type"], c["case_type"]), hit, len(ans)

        with ThreadPoolExecutor(max_workers=6) as pool:
            out = list(pool.map(_one, prepared))
        avg_hit = sum(x[2] for x in out) / len(out)
        full_pass = sum(x[2] == 1.0 for x in out) / len(out)
        avg_len = sum(x[3] for x in out) / len(out)
        records.append({"temperature": temp, "关键词平均覆盖率": round(avg_hit, 3),
                        "全断言通过率": f"{full_pass*100:.1f}%", "平均答案字数": round(avg_len)})
        print(f"[temperature={temp}] 关键词覆盖率={avg_hit:.3f} 全通过={full_pass*100:.1f}% "
              f"平均字数={avg_len:.0f}")
    return pd.DataFrame(records)


# ================================================================ 报告
def _stats(rows: list[dict]) -> dict:
    total = len(rows)
    passed = sum(r["ok"] for r in rows)
    by_type: dict[str, list] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r["ok"])
    return {
        "总准确率": f"{passed}/{total} = {passed/total*100:.1f}%",
        **{f"{TYPE_LABELS.get(t, t)}通过率": f"{sum(v)}/{len(v)} = {sum(v)/len(v)*100:.0f}%"
           for t, v in by_type.items()},
    }


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60, "\n① 第0轮：纯大模型无知识库基线（60条）\n" + "=" * 60)
    r0 = run_round0_baseline()
    print(json.dumps(_stats(r0), ensure_ascii=False, indent=2))

    print("\n" + "=" * 60, "\n② 第4轮：在线 ReAct Agent（60条）\n" + "=" * 60)
    r4 = run_round4_online()
    print(json.dumps(_stats(r4), ensure_ascii=False, indent=2))

    print("\n" + "=" * 60, "\n③ temperature 消融（12条代表题）\n" + "=" * 60)
    df_temp = run_temperature_ablation()

    # ---- 失败用例打印原始回答（人工审计用）
    for name, rows in (("第0轮", r0), ("第4轮在线", r4)):
        fails = [r for r in rows if not r["ok"]]
        print(f"\n----- {name} 未通过断言 {len(fails)} 条（以下为原始回答，请人工复核）-----")
        for r in fails:
            print(f"#{r['id']}[{TYPE_LABELS.get(r['type'], r['type'])}] {r['question'][:30]}")
            print(f"  归因: {r['reason']}")
            print(f"  回答: {r['answer'][:180]}")

    # ---- 落盘
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    xlsx = EVAL_DIR / f"online_eval_{ts}.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        pd.DataFrame(r0).to_excel(w, sheet_name="round0_pure_llm", index=False)
        pd.DataFrame(r4).to_excel(w, sheet_name="round4_online_agent", index=False)
        df_temp.to_excel(w, sheet_name="temperature_ablation", index=False)
    print(f"\n原始回答已落盘：{xlsx}")


if __name__ == "__main__":
    main()
