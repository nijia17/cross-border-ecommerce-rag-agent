# -*- coding: utf-8 -*-
"""
步骤 3：自动化评测
=================
运行 60 条分层用例（常规/口语/工具/拒答/边界/对抗/跨语言/多跳/否定），
对比三个迭代轮次准确率，输出 Markdown 报告与明细 Excel，
并同步生成 eval/test_cases.xlsx 与 promptfooconfig.yaml。

用法：
  python run_eval.py              # 三/四轮迭代对比（离线路由 + 检索，任何环境可跑）
  python run_eval.py --baseline   # 额外实测“第0轮纯大模型基线”（需在 .env 配置 DeepSeek）
"""
import json
import sys

from src.evaluation import baseline_llm_round, evaluate

if __name__ == "__main__":
    result = evaluate()
    print("=" * 60)
    print("评测完成")
    print("=" * 60)
    print(json.dumps(result["stage_metrics"], ensure_ascii=False, indent=2))
    print("Badcase 归因：", json.dumps(result["badcase归因"], ensure_ascii=False))
    print("报告文件：", result["report_md"])
    print("明细文件：", result["report_xlsx"])

    if "--baseline" in sys.argv:
        print("\n—— 第 0 轮：纯大模型无知识库基线 ——")
        baseline = baseline_llm_round()
        if baseline is None:
            print("当前为离线演示模式（未配置 DEEPSEEK_API_KEY），无法运行纯大模型基线；")
            print("配置 .env 后重新执行 python run_eval.py --baseline 即可实测 55% 基线口径。")
        else:
            failures = baseline.pop("failures")
            print(json.dumps(baseline, ensure_ascii=False, indent=2))
            for f in failures:
                print(" -", f)
