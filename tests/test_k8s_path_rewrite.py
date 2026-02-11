"""Tests for K8s path rewrite helpers used by CLI configuration rewriting."""

from __future__ import annotations

from main import _rewrite_config_paths_for_k8s, _rewrite_path_for_k8s


def test_rewrite_path_for_k8s_handles_relative_workspace_and_lance_prefixes():
    """Verify path rewriting across relative/absolute/lance-prefixed variants.

    Inputs/outputs:
        Uses representative local paths and asserts expected `s3a://` rewrites.

    Side effects:
        None.

    Assumptions:
        Known local prefixes map to `sample/` and `output/` object paths.
    """
    # NOTE(readability): This test protects path normalization contracts used by
    # in-cluster job submission, where local container paths must become S3 URIs.
    bucket = "kaiyuan-ray"
    assert (
        _rewrite_path_for_k8s("data/sample/example.lance", bucket)
        == "s3a://kaiyuan-ray/sample/example.lance"
    )
    assert (
        _rewrite_path_for_k8s("workspace/data/sample/example.lance", bucket)
        == "s3a://kaiyuan-ray/sample/example.lance"
    )
    assert (
        _rewrite_path_for_k8s("/workspace/data/output/out.lance", bucket)
        == "s3a://kaiyuan-ray/output/out.lance"
    )
    assert (
        _rewrite_path_for_k8s("lance.data/sample/example.lance", bucket)
        == "s3a://kaiyuan-ray/sample/example.lance"
    )
    assert (
        _rewrite_path_for_k8s("lance.`/workspace/data/sample/example.lance`", bucket)
        == "s3a://kaiyuan-ray/sample/example.lance"
    )


def test_rewrite_tiny_seq_classifier_to_zip():
    """Ensure tiny sequence-classifier directory paths rewrite to zip artifact.

    Inputs/outputs:
        Provides model directory paths and asserts zip URI mapping.

    Side effects:
        None.

    Assumptions:
        Model is distributed as one zip archive in object storage.
    """
    # NOTE(readability): Both root directory and nested file references should
    # map to the same zip URI to keep model loading logic simple.
    bucket = "kaiyuan-ray"
    assert (
        _rewrite_path_for_k8s("data/models/tiny_seq_classifier", bucket)
        == "s3a://kaiyuan-ray/models/tiny_seq_classifier.zip"
    )
    assert (
        _rewrite_path_for_k8s("workspace/data/models/tiny_seq_classifier/config.json", bucket)
        == "s3a://kaiyuan-ray/models/tiny_seq_classifier.zip"
    )


def test_recursive_config_rewrite_preserves_non_paths():
    """Validate recursive rewrite updates paths but preserves scalar metadata.

    Inputs/outputs:
        Rewrites nested config dict and asserts path/scalar expectations.

    Side effects:
        None.

    Assumptions:
        Non-path strings under unrelated config keys must remain unchanged.
    """
    source = {
        "ray": {"app_name": "demo"},
        "pipeline": {
            "type": "LanceWriter",
            "output_path": "data/output/out.lance",
            "child_configs": [
                {
                    "type": "LanceReader",
                    "input_path": "data/sample/in.lance",
                }
            ],
        },
    }
    rewritten = _rewrite_config_paths_for_k8s(source, "bucket-a")
    assert rewritten["ray"]["app_name"] == "demo"
    assert rewritten["pipeline"]["output_path"] == "s3a://bucket-a/output/out.lance"
    assert (
        rewritten["pipeline"]["child_configs"][0]["input_path"]
        == "s3a://bucket-a/sample/in.lance"
    )
