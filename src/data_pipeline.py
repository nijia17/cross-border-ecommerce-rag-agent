# -*- coding: utf-8 -*-
"""
知识库结构化治理管线（对应项目文档第三章）
=========================================
1) 由业务种子库展开出 519 条高质量问答（10 大分类）；
2) 注入重复/不规范/测试/过时/脏数据，构造出 567 条“原始素材”；
3) 执行与文档一致的四步清洗：去重合并 → 无效剔除 → 标准化 → 复核，
   还原 519 条结构化知识库；
4) 输出 knowledge_base_raw.xlsx / knowledge_base.xlsx 与清洗报告 JSON。

运行方式： python prepare_data.py
"""
from __future__ import annotations

import json
import random
import re
from datetime import date, timedelta

import pandas as pd

from config import (
    CATEGORIES,
    DATA_DIR,
    KB_FILE,
    RAW_KB_FILE,
)
from src.kb_seeds import SEEDS, expected_counts

RAW_TOTAL = 567
CLEAN_TOTAL = 519
RAW_COLUMNS = ["问题", "标准答案", "业务分类", "更新时间"]
CLEAN_COLUMNS = ["问题", "标准答案", "业务分类", "适用地区", "更新时间"]

# 早于该日期的内容视为“过时政策”，清洗时剔除
OUTDATED_BEFORE = date(2023, 1, 1)


# ---------------------------------------------------------------- 1. 生成 519 条高质量数据
def _random_date(start: date, end: date, rng: random.Random) -> str:
    delta = (end - start).days
    return (start + timedelta(days=rng.randint(0, delta))).isoformat()


def build_clean_records() -> list[dict]:
    """把种子模板展开为带元数据的高质量问答记录。"""
    rng = random.Random(20260101)
    records: list[dict] = []
    for category, seeds in SEEDS.items():
        cat_rows: list[dict] = []
        for seed in seeds:
            if len(seed) == 2:
                question, answer = seed
                variants = [{}]
            else:
                question, answer, variants = seed
            for slot in variants:
                region = slot.get("_region", "通用")
                slot = {k: v for k, v in slot.items() if not k.startswith("_")}
                cat_rows.append(
                    {
                        "问题": question.format(**slot),
                        "标准答案": answer.format(**slot),
                        "业务分类": category,
                        "适用地区": region,
                        "更新时间": _random_date(date(2024, 6, 1), date(2026, 9, 30), rng),
                    }
                )
        target = expected_counts()[category]
        assert len(cat_rows) >= target, f"{category} 仅生成 {len(cat_rows)} 条，不足 {target}"
        if len(cat_rows) > target:
            cat_rows = cat_rows[:target]
        records.extend(cat_rows)

    assert len(records) == CLEAN_TOTAL, f"高质量数据应为 {CLEAN_TOTAL} 条，实际 {len(records)}"
    return records


# ---------------------------------------------------------------- 2. 构造 567 条原始素材
def build_raw_records(clean: list[dict]) -> list[dict]:
    """在干净数据上注入 48 条典型脏数据，模拟真实采集的 567 条原始素材。"""
    rng = random.Random(20261023)
    raw = [
        {
            "问题": r["问题"],
            "标准答案": r["标准答案"],
            "业务分类": r["业务分类"],
            "更新时间": r["更新时间"],
        }
        for r in clean
    ]
    noise: list[dict] = []

    # (a) 完全重复数据：运营多人录入导致（20 条）
    for idx in [3, 17, 42, 88, 105, 156, 210, 267, 312, 355,
                401, 438, 470, 500, 25, 66, 130, 233, 377, 499]:
        src = clean[idx]
        noise.append({
            "问题": src["问题"],
            "标准答案": src["标准答案"],
            "业务分类": src["业务分类"],
            "更新时间": src["更新时间"],
        })

    # (b) 表述形式不统一：标点全半角/语气符号差异，标准化保留、归一化后应合并（8 条）
    for i, idx in enumerate([8, 55, 120, 188, 245, 301, 410, 466]):
        src = clean[idx]
        base = src["问题"]
        if i % 2 == 0:
            q = base[:-1] + "？？"                    # 重复问号
        elif "，" in base:
            q = base.replace("，", ",", 1)           # 半角逗号
        else:
            q = base[:-1] + "~？"                    # 多余语气符号
        noise.append({
            "问题": q,
            "标准答案": "答：" + src["标准答案"],   # 答案前缀不统一
            "业务分类": " " + src["业务分类"] + " ",
            "更新时间": src["更新时间"],
        })

    # (c) 测试 / 无效数据（10 条）
    for i in range(10):
        noise.append({
            "问题": f"测试数据请忽略{i:03d}",
            "标准答案": "test / 无",
            "业务分类": "测试" if i % 2 == 0 else "临时",
            "更新时间": _random_date(date(2025, 1, 1), date(2026, 6, 30), rng),
        })

    # (d) 过时政策内容（6 条）
    outdated = [
        ("疫情期间发货时效是多久？", "【已过时】受疫情影响，全球物流暂停，发货时间待定。", "物流配送"),
        ("英国脱欧前包裹还按欧盟政策清关吗？", "【已废止】英国仍在欧盟关税同盟内，无需额外清关。", "关税税费"),
        ("欧盟22欧元以下包裹免税吗？", "【已过时】欧盟22欧元以下小包免征增值税。", "关税税费"),
        ("旧版支付通道支持西联汇款吗？", "【已停用】支持西联汇款线下打款下单。", "支付方式"),
        ("旧版售后政策支持30天无理由退货吗？", "【已过时】所有商品支持30天无理由退货。", "退换货政策"),
        ("疫情期间可以申请延期付款吗？", "【已过时】疫情期间所有订单可延期60天付款。", "支付方式"),
    ]
    for q, a, c in outdated:
        noise.append({
            "问题": q, "标准答案": a, "业务分类": c,
            "更新时间": _random_date(date(2019, 1, 1), date(2021, 12, 31), rng),
        })

    # (e) 脏数据：空答案 / 无意义问题（4 条）
    for q, a in [("啊", ""), ("12345", "不知道"), ("。。。", ""), ("xxx", "N/A")]:
        noise.append({
            "问题": q, "标准答案": a, "业务分类": "常见问题",
            "更新时间": _random_date(date(2025, 1, 1), date(2026, 6, 30), rng),
        })

    assert len(noise) == RAW_TOTAL - CLEAN_TOTAL, f"脏数据应为 {RAW_TOTAL - CLEAN_TOTAL} 条"
    raw.extend(noise)
    rng.shuffle(raw)
    assert len(raw) == RAW_TOTAL
    return raw


# ---------------------------------------------------------------- 3. 清洗治理
_PUNCT_RE = re.compile(r"[\s\u3000，。！？!?；：、,.~～\-—_（）()【】\[\]\"'`·:;]+")
_TEST_RE = re.compile(r"(测试|test|请忽略|n/a|xxx)", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """归一化文本：去空白、统一标点、转小写，用于去重键计算。"""
    text = str(text).strip()
    text = text.replace("?", "？").replace("!", "！")
    text = _PUNCT_RE.sub("", text)
    return text.lower()


def standardize_row(row: dict) -> dict:
    """标准化处理：去除多余空白、统一答案前缀与分类标签。"""
    question = re.sub(r"\s+", "", str(row["问题"]).strip())
    question = question.replace("?", "？")
    answer = str(row["标准答案"]).strip()
    answer = re.sub(r"^答[:：]", "", answer)
    answer = re.sub(r"\s+", " ", answer)
    category = str(row["业务分类"]).strip()
    return {
        "问题": question,
        "标准答案": answer,
        "业务分类": category,
        "更新时间": str(row["更新时间"]).strip(),
    }


def clean_records(raw: list[dict]) -> tuple[list[dict], dict]:
    """四步清洗，返回（干净记录, 清洗统计）。采用两遍扫描，统计结果不受行序影响。"""
    stats = {"原始条数": len(raw), "完全重复": 0, "表述重复合并": 0,
             "测试无效数据": 0, "过时政策": 0, "空/脏内容": 0, "非标准分类": 0}

    # ---- 第一遍：标准化 + 无效内容/非法分类剔除
    valid_rows: list[dict] = []
    for raw_row in raw:
        row = standardize_row(raw_row)
        if len(row["问题"]) < 6 or len(row["标准答案"]) < 4:
            stats["空/脏内容"] += 1
            continue
        if _TEST_RE.search(row["问题"]) or _TEST_RE.fullmatch(row["标准答案"] or ""):
            stats["测试无效数据"] += 1
            continue
        if any(flag in row["标准答案"] for flag in ("【已过时】", "【已废止】", "【已停用】")):
            stats["过时政策"] += 1
            continue
        if row["业务分类"] not in CATEGORIES:
            stats["非标准分类"] += 1
            continue
        valid_rows.append(row)

    # ---- 第二遍：先按原始文本精确去重，再按归一化键合并表述差异
    cleaned: list[dict] = []
    seen_raw: set[str] = set()
    seen_norm: set[str] = set()
    for row in valid_rows:
        raw_key = row["问题"]                       # 标准化（去空白）后的原文
        norm_key = normalize_text(row["问题"])      # 再忽略标点/大小写后的归一化键
        if raw_key in seen_raw:
            stats["完全重复"] += 1                  # 完全重复问答 → 删除
            continue
        if norm_key in seen_norm:
            stats["表述重复合并"] += 1              # 表述不同、口径一致 → 合并
            continue
        seen_raw.add(raw_key)
        seen_norm.add(norm_key)

        # ---- 标准化补充“适用地区”标签
        m = re.search(r"(美国|英国|德国|法国|意大利|西班牙|澳大利亚|日本|韩国|加拿大|俄罗斯|新加坡|马来西亚|泰国|阿联酋|巴西|欧盟)", row["问题"])
        row["适用地区"] = m.group(1) if m else "通用"
        cleaned.append(row)

    stats["清洗后条数"] = len(cleaned)
    return cleaned, stats


# ---------------------------------------------------------------- 4. 落盘
def save_excel(rows: list[dict], path, columns: list[str]) -> None:
    df = pd.DataFrame(rows)[columns]
    df.to_excel(path, index=False)


def run() -> dict:
    """端到端执行：519 高质量数据 → 567 原始素材 → 清洗还原 519。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean_seed = build_clean_records()
    raw = build_raw_records(clean_seed)
    save_excel(raw, RAW_KB_FILE, RAW_COLUMNS)

    cleaned, stats = clean_records(raw)
    save_excel(cleaned, KB_FILE, CLEAN_COLUMNS)

    # 分类分布统计
    dist = pd.DataFrame(cleaned)["业务分类"].value_counts().reindex(CATEGORIES).to_dict()
    stats["分类分布"] = dist
    report_path = DATA_DIR / "cleaning_report.json"
    report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, ensure_ascii=False, indent=2))
