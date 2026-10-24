# -*- coding: utf-8 -*-
"""
跨境电商 RAG 智能问答 Agent —— 交互式命令行入口
==============================================
用法： python app.py
命令： /debug  切换调试信息（粗筛分类/得分/检索轨迹）
      /quit   退出
"""
from __future__ import annotations

from config import LLM_AVAILABLE
from src.qa_chain import smart_answer


def main() -> None:
    mode = "在线大模型（ReAct Agent）" if LLM_AVAILABLE else "离线演示模式（规则工具 + 知识库检索，配置 .env 后启用 DeepSeek）"
    print("=" * 66)
    print("  跨境电商 RAG 智能问答 Agent")
    print(f"  运行模式：{mode}")
    print("  输入问题开始咨询； /debug 切换调试信息； /quit 退出")
    print("=" * 66)

    debug = False
    while True:
        try:
            query = input("\n我： ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not query:
            continue
        if query == "/quit":
            print("再见！")
            break
        if query == "/debug":
            debug = not debug
            print(f"[调试信息已{'开启' if debug else '关闭'}]")
            continue

        result = smart_answer(query)
        print(f"\n客服：{result['answer']}")
        if debug:
            print(f"  [via={result.get('via')} 分类={result.get('category')} "
                  f"置信度={result.get('top_score')}]")
            for i, t in enumerate(result.get("traces", []), 1):
                print(f"  [top{i}] final={t['final']} sem={t['semantic']} "
                      f"kw={t['keyword']} 《{t['category']}》{t['question']}")


if __name__ == "__main__":
    main()
