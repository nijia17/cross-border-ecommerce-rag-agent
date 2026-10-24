# -*- coding: utf-8 -*-
"""
5 类业务工具 + 知识库检索工具（对应文档第五章）
=============================================
1. exchange_rate_calculator  汇率换算（内置基准汇率模拟计算）
2. content_compliance_check  文案合规检测（广告法极限词/虚假承诺关键词匹配）
3. translator                多语种翻译（有大模型走大模型，离线走内置客服词库）
4. order_delivery_query      订单时效查询（订单号规则模拟，返回标准物流节点）
5. tax_estimator             税费估算（按品类税率 + 目的国规则模拟计算）
6. knowledge_base_qa         知识库三段式检索（供 Agent 调用）
"""
from __future__ import annotations

import hashlib
import re

from langchain_core.tools import tool

from config import FALLBACK_REPLY
from src.llm import get_llm
from src.prompts import TRANSLATE_PROMPT
from src.retriever import enhanced_retrieval


# ============================================================ 1. 汇率换算
_RATE_MAP = {"USD": 7.20, "EUR": 7.80, "GBP": 9.10, "JPY": 0.048,
             "AUD": 4.70, "CAD": 5.30, "HKD": 0.92, "SGD": 5.35}
_CURRENCY_CN = {"美元": "USD", "欧元": "EUR", "英镑": "GBP", "日元": "JPY",
                "澳元": "AUD", "加元": "CAD", "港币": "HKD", "新币": "SGD"}


@tool
def exchange_rate_calculator(amount: float, currency: str, direction: str = "to_cny") -> str:
    """汇率换算工具。amount=金额，currency=外币币种代码或中文名（支持 USD美元/EUR欧元/GBP英镑/JPY日元/AUD澳元/CAD加元/HKD港币/SGD新币），direction=to_cny 表示外币换人民币（默认），from_cny 表示人民币换外币。返回换算结果（基准汇率仅供参考）。"""
    code = _CURRENCY_CN.get(currency, currency.upper())
    if code not in _RATE_MAP:
        return f"暂不支持 {currency} 币种换算，当前支持：{', '.join(_CURRENCY_CN)}（或 USD/EUR/GBP/JPY 等代码）"
    rate = _RATE_MAP[code]
    if direction == "from_cny":
        result = amount / rate
        return f"{amount:.2f} 人民币 ≈ {result:.2f} {code}（基准汇率 1 {code}={rate} CNY，仅供参考，以银行实际牌价为准）"
    cny = amount * rate
    return f"{amount:.2f} {code} ≈ {cny:.2f} 人民币（基准汇率 1 {code}={rate} CNY，仅供参考，以银行实际牌价为准）"


# ============================================================ 2. 文案合规检测
_FORBIDDEN_WORDS = {
    "广告法极限词": ["最佳", "最好", "最优", "最高级", "最低", "最便宜", "最先进", "最",
                  "第一", "国家级", "世界级", "顶级", "极品", "绝对", "万能", "永久",
                  "100%", "百分之百", "百分百", "独一无二", "首选", "之王", "史上最", "全网第一"],
    "虚假/无法证实承诺": ["无效退款", "包治百病", "根治", "永不复发", "永不反弹", "假一赔万", "零风险"],
    "普通商品医疗宣称": ["治疗", "疗效", "抗癌", "消炎", "治病", "医用级"],
}


@tool
def content_compliance_check(text: str) -> str:
    """文案合规检测工具。输入商品营销文案原文，返回命中的广告法极限词/违规风险词及修改建议；未命中则提示初步检测合规。跨境商品上架文案审核场景使用。"""
    hits: list[str] = []
    for risk_type, words in _FORBIDDEN_WORDS.items():
        hit = [w for w in words if w in text]
        if hit:
            hits.append(f"【{risk_type}】{', '.join(hit)}")
    if hits:
        return (
            "检测结果：存在合规风险，建议修改后再上架。\n- "
            + "\n- ".join(hits)
            + "\n修改建议：删除极限词和绝对化表述，用可证实的客观描述替代"
            "（如“热销款”替代“第一”，“很多顾客回购”替代“100%好评”）。"
        )
    return "检测结果：未命中极限词与高风险词库，文案初步检测合规；建议结合目的国平台规则（如欧盟GPSR、广告法）二次确认。"


# ============================================================ 3. 多语种翻译
_OFFLINE_PHRASES = {
    "感谢您的购买": {"en": "Thank you for your purchase.", "ja": "ご購入ありがとうございます。"},
    "您的包裹已发货": {"en": "Your order has been shipped.", "ja": "商品を発送いたしました。"},
    "请提供您的订单号": {"en": "Please provide your order number.", "ja": "注文番号をお知らせください。"},
    "抱歉给您带来不便": {"en": "We are sorry for the inconvenience.", "ja": "ご不便をおかけして申し訳ございません。"},
    "您的退款已处理": {"en": "Your refund has been processed.", "ja": "返金手続きが完了いたしました。"},
    "物流信息正在更新": {"en": "Tracking information is being updated.", "ja": "配送情報を更新中です。"},
    "欢迎再次光临": {"en": "We look forward to seeing you again.", "ja": "またのご来店をお待ちしております。"},
    "请耐心等待清关": {"en": "Please wait patiently for customs clearance.", "ja": "通関手続きまで今しばらくお待ちください。"},
}
_LANG_CODE = {"英文": "en", "英语": "en", "en": "en", "英文语": "en",
              "日文": "ja", "日语": "ja", "ja": "ja"}


@tool
def translator(text: str, target_lang: str = "en") -> str:
    """多语种翻译工具，适配跨境客服场景。text=待翻译内容，target_lang=目标语言（英文/英语/en、日文/日语/ja，默认英文）。输出译文。"""
    lang = _LANG_CODE.get(target_lang.lower().strip(), "en")

    llm = get_llm()
    if llm is not None:
        prompt = TRANSLATE_PROMPT.format(
            target_lang="英文" if lang == "en" else "日文", text=text
        )
        return llm.invoke(prompt).content.strip()

    # 离线降级：内置客服高频短句
    for phrase, mapping in _OFFLINE_PHRASES.items():
        if phrase in text:
            return mapping.get(lang, mapping["en"])
    return (
        "【离线演示模式】当前未配置大模型，仅支持内置客服短句互译"
        "（如“感谢您的购买”“您的包裹已发货”等）；配置 DEEPSEEK_API_KEY 后可翻译任意内容。"
    )


# ============================================================ 4. 订单时效查询
_ORDER_STAGES = [
    ("已下单/待发货", "仓库正在拣货打包，预计24小时内出库。"),
    ("已出库/干线运输", "包裹已交承运商，正在国际干线运输中，请耐心等待。"),
    ("到达目的国/清关中", "包裹正在办理进口清关，正常1-3个工作日；如收到补税通知请及时处理。"),
    ("清关完成/本地派送", "清关已完成，本地物流商正在派送，一般1-3个工作日签收。"),
    ("已签收", "包裹已完成签收。如有问题请在48小时内联系客服。"),
]


@tool
def order_delivery_query(order_id: str) -> str:
    """订单物流时效查询工具。输入订单号或运单号（字母+数字组合，6-20位，如 CB20261023A88），返回模拟的当前物流节点、状态说明和预计时效。无法识别的单号返回格式提示。"""
    oid = order_id.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{6,20}", oid):
        return "订单号格式不正确，请提供6-20位字母或数字组合的订单号（如 CB20261023A88）。"
    idx = int(hashlib.md5(oid.encode()).hexdigest(), 16) % len(_ORDER_STAGES)
    stage, desc = _ORDER_STAGES[idx]
    eta = ["约1-2天", "约5-10天", "约1-3天", "约1-3天", "已送达"][idx]
    return f"订单 {oid} 当前状态：【{stage}】\n情况说明：{desc}\n预计还需时效：{eta}（模拟数据，仅供参考，实际以物流轨迹为准）"


# ============================================================ 5. 税费估算
# 跨境电商零售进口综合税率（增值税按70%计征；含消费税品类为折算综合税率）
_TAX_RATE_TABLE = {
    "服装": 0.091, "服饰": 0.091, "鞋靴": 0.091, "箱包": 0.091,
    "母婴": 0.091, "玩具": 0.091, "食品": 0.091, "保健品": 0.091,
    "数码": 0.091, "3c": 0.091, "手机": 0.091, "电脑": 0.091, "家电": 0.091,
    "化妆品": 0.091, "护肤品": 0.091,
    "彩妆": 0.2305, "香水": 0.2305, "美容修饰": 0.2305,
    "酒水": 0.28, "手表": 0.091, "首饰": 0.091,
}
_DESTINATION_RULES = {
    "中国": "跨境电商零售进口：单次限值5000元、年度限值26000元内按综合税率征税；超限值按一般贸易方式征税。",
    "美国": "美国通常对800美元/人/天以下合规包裹免税（Section 321），超出部分按HS编码税率征收，以海关核定为准。",
    "欧盟": "欧盟无低值VAT免税，所有包裹均需缴纳增值税（标准税率约17%-27%），关税起征点为150欧元。",
    "英国": "135英镑以下包裹由销售方代收20%VAT；超过135英镑另需缴纳进口关税。",
    "日本": "完税价格1万日元以下通常免税；超出部分征收约10%消费税及按HS编码的关税。",
    "澳大利亚": "1000澳元以下包裹由销售方代收10%GST；超出按正式进口规则征税。",
}


@tool
def tax_estimator(category: str, value: float, country: str = "中国") -> str:
    """税费估算工具。category=商品品类（如服装、数码、护肤品、彩妆香水、保健品、酒水等），value=申报价值（人民币元，跨境电商进口口径），country=目的国（中国/美国/欧盟/英国/日本/澳大利亚，默认中国）。返回估算税费与规则说明，仅供参考。"""
    if value <= 0:
        return "申报价值需为大于0的数字。"
    cat = category.lower().strip()
    rate = next((r for k, r in _TAX_RATE_TABLE.items() if k in cat), None)
    rule = next((v for k, v in _DESTINATION_RULES.items() if k in country), _DESTINATION_RULES["中国"])

    if rate is None or "中国" not in country:
        return (
            f"目的国规则：{rule}\n"
            f"申报价值约 {value:.2f} 元，精确税费需结合商品HS编码、申报币种和海关审价确定，"
            "建议下单前在结算页查看含税到手价。本结果仅供参考。"
        )

    if value > 5000:
        return f"申报价值 {value:.2f} 元已超过跨境电商单次5000元限值，{rule} 本估算不适用于一般贸易渠道。"
    tax = value * rate
    return (
        f"商品品类：{category}（综合税率约 {rate*100:.1f}%）\n"
        f"申报价值：{value:.2f} 元\n"
        f"估算税费：约 {tax:.2f} 元，含税到手约 {value + tax:.2f} 元。\n"
        f"规则说明：{rule}\n（模拟估算，仅供参考，实际以海关核定为准）"
    )


# ============================================================ 6. 知识库检索
@tool
def knowledge_base_qa(query: str) -> str:
    """跨境电商知识库检索工具。输入用户的业务问题（物流、关税、退换货、支付、商品、订单、账号、合规、活动、客服等），返回知识库中最相关的问答资料；知识库没有相关内容时返回拒答提示。回答业务问题前应优先调用本工具。"""
    result = enhanced_retrieval(query)
    if not result.accepted:
        return result.reject_message or FALLBACK_REPLY
    blocks = []
    for doc in result.documents:
        q = doc.metadata.get("question", "")
        # 去掉拼接前缀“【分类】问题\n”，只保留答案正文作为工具输出
        content = doc.page_content
        answer = content.split("\n", 1)[1] if "\n" in content else content
        blocks.append(f"问：{q}\n答：{answer}")
    return "\n---\n".join(blocks)


# 工具汇总（顺序即 Agent 可选工具顺序）
BUSINESS_TOOLS = [
    knowledge_base_qa,
    exchange_rate_calculator,
    content_compliance_check,
    translator,
    order_delivery_query,
    tax_estimator,
]
