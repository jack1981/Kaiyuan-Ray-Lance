#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regexp/clean_rules.py

定义用于清洗文本的正则表达式规则（替换模式）
这些规则会查找目标字符串并将其替换为空格，但保留数据集中的序列
"""

import re
from typing import List, Tuple


class CleanRule:
    """单个清洗规则"""

    def __init__(
        self,
        pattern: str,
        replacement: str = " ",
        description: str = "",
        flags: int = 0,
    ):
        self.pattern = re.compile(pattern, flags=flags)
        self.replacement = replacement
        self.description = description

    def apply(self, text: str) -> str:
        """应用清洗规则"""
        return self.pattern.sub(self.replacement, text)


# ---------------------------
# 清洗规则定义
# ---------------------------

CLEAN_RULES = [
    # 1. HTML 标签和实体
    CleanRule(pattern=r"<[^>]+>", description="HTML tags"),
    CleanRule(pattern=r"&[a-zA-Z]+;|&#\d+;", description="HTML entities"),
    # 2. URL 和 Email
    CleanRule(pattern=r"https?://[^\s]+|www\.[^\s]+|ftp://[^\s]+", description="URLs"),
    CleanRule(
        pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        description="Email addresses",
    ),
    # 3. Markdown 图片格式
    CleanRule(pattern=r"!?\[[^\]]*\]\([^)]+\)", description="Markdown images"),
    # 4. 分页相关内容
    CleanRule(pattern=r"\d+\s*页|\d+/\d+页?", description="Page numbers"),
    CleanRule(
        pattern=r"[上下首尾末]\s*页|转到|跳转|页码|共\s*\d+\s*页",
        description="Pagination words",
        flags=re.IGNORECASE,
    ),
    CleanRule(
        pattern=r"上一[页篇章节]|下一[页篇章节]|[<>《》]{1,2}",
        description="Previous/Next navigation",
    ),
    CleanRule(
        pattern=r"\d+\s*篇信息|信息/页|记录/页|条/页", description="Page info text"
    ),
    # 5. 导航和面包屑
    CleanRule(
        pattern=r"当前位置[:：\s]*|首页\s*[>›》]\s*|[>›》]\s*正文",
        description="Breadcrumb navigation",
        flags=re.IGNORECASE,
    ),
    # 6. 网站模板文字
    CleanRule(
        pattern=r"版权所有|免责声明|联系我们|关于我们|站点地图|网站导航|"
        r"相关[文章链接推荐]|[热门最新]文章|阅读[排行榜次数]|"
        r"[点击查看]详情|更多内容|加载中|正在加载|请稍候",
        description="Template text",
        flags=re.IGNORECASE,
    ),
    # 7. 版权和转载声明
    CleanRule(
        pattern=r"注[:：]\s*本文摘自|转载请注明|版权说明|版权归.*?所有|"
        r"©.*?版权|下载资源版权|仅供学习使用|请支持正版",
        description="Copyright statements",
        flags=re.IGNORECASE,
    ),
    # 8. 新闻来源标注
    CleanRule(
        pattern=r"(?:科技日报|新华社|中国侨网|新华网|人民网|中国报道|央视网|光明网)"
        r".*?(?:记者|通讯员|编译报道|电|讯)\s+[^\n]{0,30}|"
        r"记者\s*[：:]\s*\S+|摄\s*[：:]\s*\S+",
        description="News source attribution",
        flags=re.IGNORECASE,
    ),
    # 9. 发布/更新时间信息
    CleanRule(
        pattern=r"发布时间\s*[：:]\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}.*?\d{1,2}\s+\d{1,2}.*?来源|"
        r"更新时间\s*[：:]\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}.*?来源|"
        r"来源\s*[：:]\s*[^\n]{0,50}(?:网|报|台|社)",
        description="Publish/update time info",
        flags=re.IGNORECASE,
    ),
    # 10. 媒体标记
    CleanRule(
        pattern=r"【[^】]*(?:亿邦动力|财经|科技|讯|电)[^】]*】",
        description="Media tags",
        flags=re.IGNORECASE,
    ),
    # 11. 人员职位列表
    CleanRule(
        pattern=r"(?:设计|管理|技术|开发|操作|销售|市场)人员[、，\s]*(?:等\.{2,})?",
        description="Personnel lists",
        flags=re.IGNORECASE,
    ),
    # 12. 选项标记
    CleanRule(
        pattern=r"[a-z]\.\s*[a-z]\.\s*[a-z]\.\s*[a-z]\.\s*(?:\(缺选项\))?",
        description="Option markers",
        flags=re.IGNORECASE,
    ),
    # 13. 频道/栏目标记
    CleanRule(
        pattern=r"(?:新华网|人民网|中国网|央视网)\s*\S+\s*频道\s*讯",
        description="Channel tags",
        flags=re.IGNORECASE,
    ),
    # 14. 时间戳格式（保留日期，去掉时分秒）
    CleanRule(pattern=r"\d{1,2}:\d{2}:\d{2}", description="Timestamps"),
    # 15. 长数字串（可能是ID、订单号等）
    CleanRule(pattern=r"\b\d{8,}\b", description="Long number strings (IDs)"),
    # 16. 连续重复字符（如"哈哈哈哈哈"缩减为"哈哈哈"）
    CleanRule(
        pattern=r"(.)\1{3,}", replacement=r"\1\1\1", description="Repeated characters"
    ),
    # 17. 特殊字符和符号（保留基本标点）
    CleanRule(
        pattern=r"[★☆■□●○◆◇▲△▼▽※→←↑↓…~`@#$%^&*_+=\[\]{}|\\;\'<>/]",
        description="Special characters",
    ),
    # 18. 广告和营销文本
    CleanRule(
        pattern=r"点击.*?了解更多|立即[购买咨询]|限时[优惠特价]|"
        r"扫码关注|关注公众号|长按.*?识别|识别二维码",
        description="Advertisement text",
        flags=re.IGNORECASE,
    ),
    # 19. 社交媒体相关
    CleanRule(
        pattern=r"微信公众号|官方微博|抖音号|小红书|B站|bilibili",
        description="Social media mentions",
        flags=re.IGNORECASE,
    ),
    # 20. 客服和联系信息
    CleanRule(
        pattern=r"客服[热线电话]*[:：\s]*\d+|咨询电话[:：\s]*\d+|"
        r"QQ[:：\s]*\d+|微信[:：\s]*[a-zA-Z0-9_]+",
        description="Customer service info",
        flags=re.IGNORECASE,
    ),
]


def get_clean_rules() -> List[CleanRule]:
    """获取所有清洗规则"""
    return CLEAN_RULES


def apply_all_clean_rules(text: str) -> str:
    """
    应用所有清洗规则到文本

    Args:
        text: 原始文本

    Returns:
        清洗后的文本
    """
    if not text or not isinstance(text, str):
        return ""

    cleaned = text

    # 应用所有规则
    for rule in CLEAN_RULES:
        cleaned = rule.apply(cleaned)

    # 统一空白字符
    cleaned = cleaned.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned
