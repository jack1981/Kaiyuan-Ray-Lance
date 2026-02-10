#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chinese_pipeline.py

Chinese text cleaning and filtering node for Ray Data.
"""

from typing import Optional

import pandas as pd

from datafiner.base import PipelineNode
from datafiner.dataset_utils import map_batches_tuned
from datafiner.regexp.clean_rules import apply_all_clean_rules
from datafiner.regexp.filter_rules import should_filter_text
from datafiner.register import register


@register("ChineseCleanAndFilter")
class ChineseCleanAndFilter(PipelineNode):
    """
    Clean and filter Chinese text.
    """

    def __init__(
        self,
        runtime,
        input_col: str = "text",
        output_col: str = "clean_text",
        child_configs: Optional[list] = None,
        min_length: int = 10,
    ):
        super().__init__(runtime, child_configs)
        self.input_col = input_col
        self.output_col = output_col
        self.min_length = min_length

    def run(self):
        if not self.children:
            raise ValueError(
                "ChineseCleanAndFilter must have at least one child node producing a dataset."
            )

        ds = self.children[0].run()

        def clean_batch(batch: pd.DataFrame) -> pd.DataFrame:
            out = batch.copy()
            texts = out[self.input_col].fillna("").astype(str)
            flags = texts.apply(should_filter_text)
            length_ok = texts.str.len() >= self.min_length
            out = out[(~flags) & length_ok]
            out[self.output_col] = out[self.input_col].apply(
                lambda x: apply_all_clean_rules(str(x)) if x is not None else None
            )
            return out

        return map_batches_tuned(ds, self.runtime, clean_batch, batch_format="pandas")
