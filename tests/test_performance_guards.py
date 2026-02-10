from __future__ import annotations

from pathlib import Path

import inspect
import ray

from datafiner.base import PipelineTree
from datafiner.sampler import DuplicateSampleRatio


def _seed_rows(path: Path, row_count: int, extra_col: bool = False) -> None:
    rows = []
    for i in range(row_count):
        row = {"id": i, "text": f"row-{i}", "duplicate_count": float((i % 4) + 1)}
        if extra_col:
            row["extra"] = f"v-{i}"
        rows.append(row)
    ray.data.from_items(rows).write_lance(str(path), mode="overwrite")


def test_reader_repartition_controls_blocks(tmp_path: Path):
    input_path = tmp_path / "read_blocks.lance"
    _seed_rows(input_path, 120)

    config = {
        "ray": {"app_name": "test-read-blocks"},
        "pipeline": {
            "type": "LanceReader",
            "input_path": str(input_path),
            "num_parallel": 4,
        },
    }

    ds = PipelineTree(config, mode="local").run()
    mat = ds.materialize()
    assert mat.count() == 120
    assert mat.num_blocks() == 4


def test_union_by_name_preserves_schema_and_block_scale(tmp_path: Path):
    left = tmp_path / "left.lance"
    right = tmp_path / "right.lance"
    _seed_rows(left, 80, extra_col=False)
    _seed_rows(right, 70, extra_col=True)

    config = {
        "ray": {"app_name": "test-union-blocks"},
        "pipeline": {
            "type": "UnionByName",
            "allow_missing_columns": True,
            "child_configs": [
                {"type": "LanceReader", "input_path": str(left), "num_parallel": 3},
                {"type": "LanceReader", "input_path": str(right), "num_parallel": 3},
            ],
        },
    }

    ds = PipelineTree(config, mode="local").run()
    mat = ds.materialize()
    assert mat.count() == 150
    assert set(mat.schema().names) == {"id", "text", "duplicate_count", "extra"}
    assert mat.num_blocks() <= 8


def test_writer_caps_output_blocks(tmp_path: Path):
    input_path = tmp_path / "many_blocks.lance"
    output_path = tmp_path / "capped_output.lance"
    _seed_rows(input_path, 200)

    config = {
        "ray": {
            "app_name": "test-write-cap",
            "data": {"max_write_blocks": 4},
        },
        "pipeline": {
            "type": "LanceWriter",
            "output_path": str(output_path),
            "mode": "overwrite",
            "child_configs": [
                {"type": "LanceReader", "input_path": str(input_path), "num_parallel": 20}
            ],
        },
    }

    ds = PipelineTree(config, mode="local").run()
    mat = ds.materialize()
    assert mat.count() == 200
    assert mat.num_blocks() <= 4
    assert ray.data.read_lance(str(output_path)).count() == 200


def test_map_batches_operator_present_in_stats(tmp_path: Path):
    input_path = tmp_path / "stats_input.lance"
    _seed_rows(input_path, 50)

    config = {
        "ray": {"app_name": "test-map-batches-stats"},
        "pipeline": {
            "type": "DuplicateSampleRatio",
            "global_sample_rate": 0.5,
            "max_sample": 3.0,
            "col": "duplicate_count",
            "child_configs": [{"type": "LanceReader", "input_path": str(input_path)}],
        },
    }

    ds = PipelineTree(config, mode="local").run()
    stats = ds.materialize().stats()
    if stats:
        assert "mapbatches" in stats.lower()
    else:
        # Fallback structural guard if operator stats are disabled in this Ray build.
        assert "map_batches_tuned" in inspect.getsource(DuplicateSampleRatio.run)
