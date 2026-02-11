"""Integration tests for transform correctness and Lance round-trip behavior."""

from __future__ import annotations

from pathlib import Path

import ray

from datafiner.base import PipelineTree


def _seed_input_dataset(path: Path) -> None:
    """Write a tiny deterministic Lance dataset used by integration tests.

    Args:
        path: Output Lance path.

    Returns:
        None.

    Side effects:
        Writes Lance dataset to local filesystem.

    Assumptions:
        Test data values remain stable for deterministic assertions.
    """
    rows = [
        {"id": 1, "text": "a", "duplicate_count": 2.0},
        {"id": 2, "text": "b", "duplicate_count": 10.0},
        {"id": 3, "text": "c", "duplicate_count": 0.0},
    ]
    ds = ray.data.from_items(rows)
    ds.write_lance(str(path), mode="overwrite")


def test_duplicate_sample_ratio_map_batches(tmp_path: Path):
    """Validate `DuplicateSampleRatio` scales and caps duplicate counts.

    Inputs/outputs:
        Runs a tiny pipeline and asserts resulting duplicate-count values.

    Side effects:
        Builds Ray pipeline and reads/writes local Lance test data.

    Assumptions:
        Transform is implemented via map-batches and preserves row count.
    """
    # NOTE(readability): Exact sorted values make the transform contract explicit
    # for future performance-oriented rewrites.
    input_path = tmp_path / "input.lance"
    _seed_input_dataset(input_path)

    config = {
        "ray": {"app_name": "test-duplicate-sample-ratio"},
        "pipeline": {
            "type": "DuplicateSampleRatio",
            "global_sample_rate": 0.5,
            "max_sample": 3.0,
            "col": "duplicate_count",
            "child_configs": [
                {
                    "type": "LanceReader",
                    "input_path": str(input_path),
                }
            ],
        },
    }

    pipeline = PipelineTree(config, mode="local")
    ds = pipeline.run()
    got = sorted(ds.to_pandas()["duplicate_count"].tolist())
    assert got == [0.0, 1.0, 3.0]


def test_lance_write_and_read_round_trip(tmp_path: Path):
    """Ensure LanceWriter produces readable output with unchanged row content.

    Inputs/outputs:
        Writes through pipeline and reads back directly via Ray Lance reader.

    Side effects:
        Writes and reads local Lance datasets.

    Assumptions:
        Basic round-trip should preserve all input rows/ids.
    """
    input_path = tmp_path / "input_rw.lance"
    output_path = tmp_path / "output_rw.lance"
    _seed_input_dataset(input_path)

    config = {
        "ray": {"app_name": "test-lance-round-trip"},
        "pipeline": {
            "type": "LanceWriter",
            "output_path": str(output_path),
            "mode": "overwrite",
            "child_configs": [
                {
                    "type": "LanceReader",
                    "input_path": str(input_path),
                }
            ],
        },
    }

    pipeline = PipelineTree(config, mode="local")
    written_ds = pipeline.run()
    assert written_ds.count() == 3

    read_back = ray.data.read_lance(str(output_path))
    assert read_back.count() == 3
    assert sorted(read_back.to_pandas()["id"].tolist()) == [1, 2, 3]
