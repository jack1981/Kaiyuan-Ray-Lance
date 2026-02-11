"""Pipeline behavior tests covering read/transform/write and writer modes."""

from __future__ import annotations

from pathlib import Path

import ray

from datafiner.base import PipelineTree


def _seed_rows(path: Path, rows: list[dict], num_blocks: int | None = None) -> None:
    """Write deterministic test rows to Lance with optional block repartitioning.

    Args:
        path: Target Lance dataset path.
        rows: Row dictionaries to write.
        num_blocks: Optional block count before write.

    Returns:
        None.

    Side effects:
        Writes local Lance files.

    Assumptions:
        Block repartitioning is non-shuffle for deterministic content order.
    """
    ds = ray.data.from_items(rows)
    if num_blocks is not None and num_blocks > 0:
        ds = ds.repartition(num_blocks, shuffle=False)
    ds.write_lance(str(path), mode="overwrite")


def test_read_transform_write_round_trip(tmp_path: Path):
    """Verify end-to-end read-transform-write pipeline preserves expected output.

    Inputs/outputs:
        Executes composed pipeline and asserts transformed values in output sink.

    Side effects:
        Reads and writes local Lance datasets.

    Assumptions:
        DuplicateSampleRatio transform should scale counts deterministically.
    """
    input_path = tmp_path / "input.lance"
    output_path = tmp_path / "output.lance"
    _seed_rows(
        input_path,
        [
            {"id": 1, "text": "a", "duplicate_count": 2.0},
            {"id": 2, "text": "b", "duplicate_count": 10.0},
            {"id": 3, "text": "c", "duplicate_count": 0.0},
        ],
    )

    config = {
        "ray": {"app_name": "test-read-transform-write"},
        "pipeline": {
            "type": "LanceWriter",
            "output_path": str(output_path),
            "mode": "overwrite",
            "child_configs": [
                {
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
                }
            ],
        },
    }

    pipeline = PipelineTree(config, mode="local")
    written = pipeline.run()
    assert written.count() == 3

    read_back = ray.data.read_lance(str(output_path))
    assert sorted(read_back.to_pandas()["duplicate_count"].tolist()) == [0.0, 1.0, 3.0]


def test_lance_writer_modes(tmp_path: Path):
    """Validate LanceWriter mode semantics across overwrite/append/ignore/create.

    Inputs/outputs:
        Runs writer in multiple modes and asserts sink row counts after each.

    Side effects:
        Repeatedly writes/reads local Lance datasets.

    Assumptions:
        `read_if_exists` should return cached dataset without recomputation.
    """
    input_a = tmp_path / "input_a.lance"
    input_b = tmp_path / "input_b.lance"
    output_path = tmp_path / "output_modes.lance"
    output_create = tmp_path / "output_create.lance"

    _seed_rows(
        input_a,
        [
            {"id": 1, "text": "a"},
            {"id": 2, "text": "b"},
            {"id": 3, "text": "c"},
        ],
    )
    _seed_rows(
        input_b,
        [
            {"id": 10, "text": "x"},
            {"id": 11, "text": "y"},
        ],
    )

    def run_writer(input_path: Path, mode: str, target: Path):
        """Build and execute a small writer pipeline for one mode.

        Args:
            input_path: Source Lance path.
            mode: Writer mode string.
            target: Destination Lance path.

        Returns:
            Output dataset from pipeline run.

        Side effects:
            Initializes and executes pipeline with Lance I/O.

        Assumptions:
            Config shape mirrors production writer config contracts.
        """
        cfg = {
            "ray": {"app_name": f"test-writer-{mode}"},
            "pipeline": {
                "type": "LanceWriter",
                "output_path": str(target),
                "mode": mode,
                "child_configs": [{"type": "LanceReader", "input_path": str(input_path)}],
            },
        }
        return PipelineTree(cfg, mode="local").run()

    overwrite_ds = run_writer(input_a, "overwrite", output_path)
    assert overwrite_ds.count() == 3
    assert ray.data.read_lance(str(output_path)).count() == 3

    append_ds = run_writer(input_b, "append", output_path)
    assert append_ds.count() == 2
    assert ray.data.read_lance(str(output_path)).count() == 5

    ignore_ds = run_writer(input_a, "ignore", output_path)
    assert ignore_ds.count() == 3
    assert ray.data.read_lance(str(output_path)).count() == 5

    cached_ds = run_writer(input_b, "read_if_exists", output_path)
    assert cached_ds.count() == 5
    assert ray.data.read_lance(str(output_path)).count() == 5

    create_ds = run_writer(input_a, "create", output_create)
    assert create_ds.count() == 3
    assert ray.data.read_lance(str(output_create)).count() == 3
