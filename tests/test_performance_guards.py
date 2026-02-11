"""Structural performance-guard tests for Ray Dataset execution plans."""

from __future__ import annotations

from pathlib import Path

import inspect
import ray

from datafiner.base import PipelineTree
from datafiner.sampler import DuplicateSampleRatio


def _seed_rows(path: Path, row_count: int, extra_col: bool = False) -> None:
    """Write synthetic rows used by structural block/operator guard tests.

    Args:
        path: Target Lance path.
        row_count: Number of rows to generate.
        extra_col: Whether to include optional schema column.

    Returns:
        None.

    Side effects:
        Writes local Lance test dataset.

    Assumptions:
        Generated values are deterministic for stable assertions.
    """
    rows = []
    for i in range(row_count):
        row = {"id": i, "text": f"row-{i}", "duplicate_count": float((i % 4) + 1)}
        if extra_col:
            row["extra"] = f"v-{i}"
        rows.append(row)
    ray.data.from_items(rows).write_lance(str(path), mode="overwrite")


def test_reader_repartition_controls_blocks(tmp_path: Path):
    """Ensure reader `num_parallel` maps to expected block count.

    Inputs/outputs:
        Runs LanceReader with repartition and asserts resulting block total.

    Side effects:
        Executes Ray read/materialize operations.

    Assumptions:
        Materialized dataset exposes accurate `num_blocks`.
    """
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
    """Ensure union-by-name preserves merged schema without block explosion.

    Inputs/outputs:
        Unions two datasets with schema mismatch and checks schema/block bounds.

    Side effects:
        Executes read/union/materialization operations.

    Assumptions:
        Missing columns should be null-filled when `allow_missing_columns=True`.
    """
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
    """Ensure writer-side block capping limits output block fan-out.

    Inputs/outputs:
        Runs write pipeline with `max_write_blocks` and asserts cap respected.

    Side effects:
        Executes read/write/materialize operations on Lance datasets.

    Assumptions:
        Cap applies when writer `num_output_files` is not explicitly set.
    """
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
    """Guard that DuplicateSampleRatio remains map-batches based.

    Inputs/outputs:
        Runs transform and checks plan stats (or source fallback) for map-batches.

    Side effects:
        Executes pipeline and materialization; introspects Python source fallback.

    Assumptions:
        Some Ray builds may omit stats text, requiring source-based fallback.
    """
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
        # NOTE(readability): Keep fallback for Ray builds that suppress operator
        # stats, so this test still enforces implementation intent.
        assert "map_batches_tuned" in inspect.getsource(DuplicateSampleRatio.run)
