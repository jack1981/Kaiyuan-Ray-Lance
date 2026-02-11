#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chinese_pipeline.py

Chinese text cleaning and filtering node for Ray Data.

This node applies regex-based spam filtering and text cleaning rules tailored
for Chinese corpora. See also `datafiner/regexp/clean_rules.py` and
`datafiner/regexp/filter_rules.py`.
It is aligned with the report's focus on preserving high-quality Chinese-domain
data in later curriculum phases.
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
        """Configure Chinese text clean/filter columns and thresholds.

        Args:
            runtime: Shared runtime config.
            input_col: Source text column.
            output_col: Destination cleaned text column.
            child_configs: Upstream node configs.
            min_length: Minimum allowed raw-text length.

        Returns:
            None.

        Side effects:
            None during initialization.

        Assumptions:
            Filtering decision is based on raw input text before cleaning.
        """
        super().__init__(runtime, child_configs)
        self.input_col = input_col
        self.output_col = output_col
        self.min_length = min_length

    def run(self):
        """Filter noisy Chinese text rows and emit cleaned text column.

        Inputs/outputs:
            Reads first child dataset and returns filtered/cleaned dataset.

        Side effects:
            Executes regex-heavy pandas batch transform.

        Assumptions:
            Node operates on first child only and requires that child to exist.
        """
        if not self.children:
            raise ValueError(
                "ChineseCleanAndFilter must have at least one child node producing a dataset."
            )

        ds = self.children[0].run()

        def clean_batch(batch: pd.DataFrame) -> pd.DataFrame:
            """Apply spam filtering, min-length checks, and cleaning rules.

            Args:
                batch: Source pandas batch.

            Returns:
                Filtered batch with `output_col` populated.

            Side effects:
                None.

            Assumptions:
                `should_filter_text` is conservative and may drop borderline rows.
            """
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
