# -*- coding: utf-8 -*-
"""
检索阈值/权重校准小工具
======================
打印一批“应命中 / 应拒答”问题的粗筛分类、最终得分和召回首条，
用于第 3-4 轮迭代时调整 SCORE_THRESHOLD、重排权重（见 config.py）。

用法： python calibrate_retrieval.py
"""
from src.retriever import enhanced_retrieval

SHOULD_HIT = [
    "美国快递一般几天能到",
    "包裹一直没更新物流",
    "关税大概要交多少",
    "衣服尺码不合适想换一件",
    "信用卡付不了款咋整",
    "怎么注销账户",
    "商品文案里写全网第一行不行",
    "优惠券过期了还给补吗",
    "人工客服几点上班",
    "增值税发票怎么开",
]
SHOULD_REJECT = [
    "请问你们老板叫什么名字",
    "今天天气怎么样",
    "我想买一只宠物猫",
    "帮我写一首七言律诗",
    "推荐几只股票",
]


def main() -> None:
    print("—— 应命中（常规/口语化问题）——")
    for q in SHOULD_HIT:
        r = enhanced_retrieval(q)
        first = r.traces[0]["question"] if r.traces else "-"
        flag = "OK " if r.accepted else "MISS"
        print(f"[{flag}] score={r.top_score:<6} cat={r.category} | {q} -> {first}")

    print("\n—— 应拒答（超纲问题）——")
    for q in SHOULD_REJECT:
        r = enhanced_retrieval(q)
        first = r.traces[0]["question"] if r.traces else "-"
        flag = "OK  " if not r.accepted else "LEAK"
        print(f"[{flag}] score={r.top_score:<6} cat={r.category} | {q} -> {first}")


if __name__ == "__main__":
    main()
