from __future__ import annotations

from main import _rewrite_config_paths_for_k8s, _rewrite_path_for_k8s


def test_rewrite_path_for_k8s_handles_relative_workspace_and_lance_prefixes():
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

