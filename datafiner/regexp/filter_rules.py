#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regexp/filter_rules.py
"""

import re
import random
from typing import List


class RandomFilterRule:
    """Single regex-based filter rule."""

    def __init__(
        self, pattern: str, keep_ratio: float, description: str = "", flags: int = 0
    ):
        self.pattern = re.compile(pattern, flags=flags)
        self.keep_ratio = keep_ratio
        self.description = description

    def should_filter(self, text: str) -> bool:
        """
        Returns True if the text should be filtered.
        """
        if self.pattern.search(text):
            decision = random.random() > self.keep_ratio
            return decision
        return False


# ---------------------------
# Tier 1: High-confidence ad / spam patterns
# ---------------------------

# 典型广告行为触发词
AD_KEYWORDS = (
    r"免费咨询|热线|扫码|点击领取|立即购买|官方客服|了解详情|拨打电话|"
    r"在线咨询|添加老师|加群|客服|加V|VX|QQ"
)

# 医疗/健康类关键词
MEDICAL_TERMS = (
    r"白癜风|癫痫|前列腺|牛皮癣|性病|肝病|肾病|痔疮|脱发|狐臭|"
    r"割包皮|早泄|阳痿|丰胸|祛斑|祛痘|隆胸|抽脂|"
    r"糖尿病|高血压|不孕不育|男科|妇科|皮肤病|增高|美白|减肥药"
)

# 改进后的微信广告模式：必须出现在广告或医疗语境中
WECHAT_AD_PATTERNS = [
    # 医疗类 + 引导加微信
    rf"(治疗|治好|偏方|医院|专家|药|医生).{{0,15}}?(加|添加|联系).{{0,5}}?微[信V]",
    # 广告关键词 + 微信
    rf"({AD_KEYWORDS}).{{0,10}}?(加|添加|联系).{{0,5}}?微[信V]",
    # 明确提供微信号的推广形式
    r"(?:微信|V信|VX|微：|v：|V：|vx)[:：]?[a-zA-Z0-9_\-]{3,}",
]

# 高置信度广告或诈骗类组合：医疗 + 广告行为
HIGH_CONF_PATTERNS = [
    rf"({AD_KEYWORDS}).*?({MEDICAL_TERMS})",
    rf"({MEDICAL_TERMS}).*?({AD_KEYWORDS})",
    *WECHAT_AD_PATTERNS,
    r"QQ[:：]?[0-9]{5,}",
    # 排除中性语境的公众号提示
    r"关注公众号[:：](?!文章|平台|功能)",
]


# ---------------------------
# Tier 2: Low-confidence medical/beauty topics
# ---------------------------

FILTER_RULES: List[RandomFilterRule] = [
    # Tier 1: 明确广告或推广，直接过滤
    *[
        RandomFilterRule(p, keep_ratio=0.05, description="High-confidence ad pattern")
        for p in HIGH_CONF_PATTERNS
    ],
    # Tier 2: 单纯医疗或美容话题
    RandomFilterRule(
        MEDICAL_TERMS,
        keep_ratio=0.1,
        description="Medical/beauty topic (probabilistic)",
    ),
]


def get_filter_rules() -> List[RandomFilterRule]:
    return FILTER_RULES


def should_filter_text(text: str) -> bool:
    """
    Returns True if text should be filtered.
    """
    if not text or not isinstance(text, str):
        return True

    for rule in FILTER_RULES:
        if rule.should_filter(text):
            return True
    return False
