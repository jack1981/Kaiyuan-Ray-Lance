#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chinese_pipeline.py

定义基于 PySpark 的中文清洗与过滤 Pipeline 节点。
该节点结合 clean_rules.py 和 filter_rules.py，
用于在 Spark DataFrame 上执行文本清洗与过滤。
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, BooleanType
from typing import Optional

from datafiner.regexp.clean_rules import apply_all_clean_rules
from datafiner.regexp.filter_rules import should_filter_text
from datafiner.base import PipelineNode
from datafiner.register import register


@register("ChineseCleanAndFilter")
class ChineseCleanAndFilter(PipelineNode):
    """
    清洗并过滤中文文本的 Pipeline 节点。

    1. 使用正则清洗文本（保持数据序列）
    2. 根据过滤规则随机丢弃部分文本
    """

    def __init__(
        self,
        spark: SparkSession,
        input_col: str = "text",
        output_col: str = "clean_text",
        child_configs: Optional[list] = None,
        min_length: int = 10,
    ):
        super().__init__(spark, child_configs)
        self.input_col = input_col
        self.output_col = output_col
        self.min_length = min_length

    def run(self) -> DataFrame:
        """
        执行清洗与过滤任务
        """
        if not self.children or len(self.children) == 0:
            raise ValueError(
                "ChineseCleanAndFilter must have at least one child node producing a DataFrame."
            )

        # 运行子节点，得到输入 DataFrame
        df = self.children[0].run()

        # 注册 UDFs
        clean_udf = F.udf(apply_all_clean_rules, StringType())
        filter_udf = F.udf(should_filter_text, BooleanType())

        # 标记过滤
        df_flagged = df.withColumn("should_filter", filter_udf(F.col(self.input_col)))

        # 应用过滤条件 + 最小长度约束
        df_filtered = df_flagged.filter(
            (F.col("should_filter") == F.lit(False))
            & (F.length(F.col(self.input_col)) >= self.min_length)
        ).drop("should_filter")
        # 清洗文本
        df_cleaned = df_filtered.withColumn(
            self.output_col, clean_udf(F.col(self.input_col))
        )

        return df_cleaned
