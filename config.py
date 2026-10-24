# -*- coding: utf-8 -*-
"""
全局配置中心
============
项目中所有路径、模型与检索参数统一在此维护，敏感信息通过项目根目录下的
 ``.env`` 文件注入（参考 ``.env.example``），代码中不允许硬编码密钥。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------- 路径配置
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EVAL_DIR = BASE_DIR / "eval"
FAISS_DIR = BASE_DIR / "faiss_index"

RAW_KB_FILE = DATA_DIR / "knowledge_base_raw.xlsx"   # 567 条原始素材
KB_FILE = DATA_DIR / "knowledge_base.xlsx"           # 清洗后 519 条高质量知识库
TEST_CASES_FILE = EVAL_DIR / "test_cases.xlsx"       # 20 条评测用例

for _d in (DATA_DIR, EVAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 加载 .env（不存在时静默跳过，进入离线演示模式）
load_dotenv(BASE_DIR / ".env", override=False)

# ---------------------------------------------------------------- 大模型配置
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "").strip().strip('"').strip("'")
DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-chat").strip()
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# 阿里云百炼（通义）在线 Embedding 可选项
DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "").strip()

# 是否配置了可用的在线大模型（未配置时整套系统进入“离线演示模式”）
LLM_AVAILABLE: bool = bool(
    DEEPSEEK_API_KEY and DEEPSEEK_API_KEY not in {"你的API密钥", "sk-xxx", "your_api_key"}
)

# ---------------------------------------------------------------- 嵌入模型配置
EMBED_DEVICE: str = os.getenv("EMBED_DEVICE", "cpu").strip()
# 默认使用文档指定的 bge-small-zh-v1.5（中文、512 维、约 95MB，可完全离线运行）
BGE_MODEL_NAME: str = os.getenv("BGE_MODEL", "BAAI/bge-small-zh-v1.5").strip()

# ---------------------------------------------------------------- 文本切分参数（对应文档 3.3）
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SPLIT_SEPARATORS = ["\n", "。", "；", "！", "？", "，", " ", ""]

# ---------------------------------------------------------------- 三段式检索参数（对应文档第四章）
RECALL_K = int(os.getenv("RECALL_K", "10"))            # 语义召回条数
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "3"))       # 重排后保留条数
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.60"))  # 置信度兜底阈值
SEMANTIC_WEIGHT = 0.7                                  # 语义得分权重
KEYWORD_WEIGHT = 0.3                                   # 关键词重合权重
CATEGORY_BOOST = 0.05                                  # 命中主题粗筛分类时的加权

# ---------------------------------------------------------------- 业务分类（10 类）
CATEGORIES = [
    "物流配送", "关税税费", "退换货政策", "支付方式", "商品信息",
    "订单查询", "账号问题", "合规要求", "活动规则", "常见问题",
]

# 主题粗筛关键词表（对应文档 4.2，已补全 10 类）
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "物流配送": ["物流", "快递", "配送", "发货", "时效", "多久到", "运单", "包裹", "运费", "派送", "签收", "清关", "仓库"],
    "关税税费": ["关税", "税费", "交税", "增值税", "清关费", "消费税", "税率", "行邮税", "综合税", "完税价格", "税"],
    "退换货政策": ["退货", "换货", "退款", "售后", "七天无理由", "7天无理由", "保修", "维修", "退回"],
    "支付方式": ["支付", "付款", "信用卡", "paypal", "微信", "支付宝", "分期", "扣款", "账单", "拒付", "币种结算"],
    "商品信息": ["尺寸", "尺码", "材质", "规格", "重量", "颜色", "库存", "成分", "正品", "保修卡", "说明书", "产地"],
    "订单查询": ["订单", "下单", "取消订单", "改单", "订单状态", "发货状态", "单号", "查一下", "物流单号"],
    "账号问题": ["账号", "账户", "登录", "注册", "密码", "验证码", "绑定", "邮箱", "手机号", "注销", "封号"],
    "合规要求": ["合规", "禁售", "限售", "认证", "ce认证", "fcc", "知识产权", "侵权", "商标", "专利", "申报", "资质",
                  "文案", "宣传", "广告词", "广告法", "极限词", "假货", "仿牌"],
    "活动规则": ["活动", "优惠", "折扣", "促销", "满减", "优惠券", "秒杀", "返现", "会员", "积分", "赠品"],
    "常见问题": ["客服", "人工", "营业时间", "发票", "联系", "怎么用", "如何", "能不能", "可以吗", "是否"],
}

# 统一拒答话术
FALLBACK_REPLY = "抱歉，暂无相关问题的解答，建议联系人工客服（服务时间 9:00-22:00）。"
