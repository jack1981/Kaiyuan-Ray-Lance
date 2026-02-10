import numpy as np
import pandas as pd

from datafiner.base import PipelineNode
from datafiner.dataset_utils import dataset_from_pandas, union_children
from datafiner.register import register


@register("DuplicateSampleRatio")
class DuplicateSampleRatio(PipelineNode):
    def __init__(
        self,
        runtime,
        child_configs: list = None,
        global_sample_rate: float = 0.1,
        max_sample: float = 20.0,
        col: str = "duplicate_count",
    ):
        super().__init__(runtime, child_configs)
        self.global_sample_rate = global_sample_rate
        self.max_sample = max_sample
        self.col = col

    def run(self):
        ds = union_children(self.children, by_name=False)

        def apply_ratio(batch: pd.DataFrame) -> pd.DataFrame:
            out = batch.copy()
            out[self.col] = (
                pd.to_numeric(out[self.col], errors="coerce").fillna(0) * self.global_sample_rate
            ).clip(upper=self.max_sample)
            return out

        return ds.map_batches(apply_ratio, batch_format="pandas")


@register("Sampler")
class Sampler(PipelineNode):
    def __init__(
        self,
        runtime,
        child_configs: list = None,
        col: str = "duplicate_count",
    ):
        super().__init__(runtime, child_configs)
        self.col = col

    def run(self):
        ds = union_children(self.children, by_name=False)
        return self.sample(ds)

    def sample(self, ds):
        def apply_sample(batch: pd.DataFrame) -> pd.DataFrame:
            out = batch.copy()
            values = pd.to_numeric(out[self.col], errors="coerce").fillna(0)
            floors = np.floor(values)
            probs = values - floors
            random_draw = np.random.rand(len(out))
            sampled = floors + (random_draw < probs)
            out[self.col] = sampled.astype(int)
            out = out[out[self.col] > 0]
            return out

        return ds.map_batches(apply_sample, batch_format="pandas")


@register("Flatten")
class Flatten(PipelineNode):
    def __init__(
        self,
        runtime,
        child_configs: list = None,
        col: str = "duplicate_count",
    ):
        super().__init__(runtime, child_configs)
        self.col = col

    def run(self):
        ds = union_children(self.children, by_name=False)
        return self.flatten(ds)

    def flatten(self, ds):
        pdf = ds.to_pandas()
        if pdf.empty:
            return ds

        repeats = pd.to_numeric(pdf[self.col], errors="coerce").fillna(0).astype(int)
        repeats = repeats.clip(lower=0)
        non_repeat_cols = [c for c in pdf.columns if c != self.col]

        expanded = pdf.loc[pdf.index.repeat(repeats)][non_repeat_cols].reset_index(drop=True)
        return dataset_from_pandas(expanded)


@register("GroupFlatten")
class GroupFlatten(PipelineNode):
    def __init__(
        self,
        runtime,
        child_configs: list = None,
        cols: list = None,
        sub_cols: list = None,
        output_cols: list = None,
    ):
        super().__init__(runtime, child_configs)
        assert cols is not None and len(cols) >= 1, (
            "GroupFlatten requires at least one array column."
        )
        self.cols = cols
        self.sub_cols = sub_cols if sub_cols is not None else cols
        self.output_cols = output_cols if output_cols is not None else self.sub_cols
        assert len(self.sub_cols) == len(self.output_cols), (
            "sub_cols and output_cols must have the same length."
        )

    def run(self):
        ds = union_children(self.children, by_name=False)
        return self.flatten(ds)

    def flatten(self, ds):
        pdf = ds.to_pandas()
        if pdf.empty:
            return ds

        expanded_rows = []
        for _, row in pdf.iterrows():
            arrays = [row.get(c) for c in self.cols]
            if not arrays or any(a is None for a in arrays):
                continue
            min_len = min(len(a) for a in arrays)
            for i in range(min_len):
                new_row = row.to_dict()
                for idx, sub_col in enumerate(self.sub_cols):
                    out_col = self.output_cols[idx]
                    source_col = self.cols[idx] if idx < len(self.cols) else self.cols[0]
                    value = row.get(source_col)
                    if isinstance(value, (list, tuple)) and i < len(value):
                        item = value[i]
                        if isinstance(item, dict):
                            new_row[out_col] = item.get(sub_col)
                        else:
                            new_row[out_col] = item
                    else:
                        new_row[out_col] = None
                expanded_rows.append(new_row)

        if not expanded_rows:
            return dataset_from_pandas(pd.DataFrame(columns=pdf.columns))

        out = pd.DataFrame(expanded_rows)
        return dataset_from_pandas(out)
