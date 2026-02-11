"""Tests for environment-derived storage options used by Lance I/O."""

from __future__ import annotations

from datafiner.dataset_utils import default_storage_options


def test_default_storage_options_preserves_scheme_and_string_allow_http(monkeypatch):
    """Ensure explicit endpoint scheme is preserved and allow-http is normalized.

    Inputs/outputs:
        Seeds AWS env vars and verifies computed storage options.

    Side effects:
        Mutates process environment via pytest `monkeypatch`.

    Assumptions:
        `LANCE_AWS_ALLOW_HTTP=1` maps to string `"true"` in options.
    """
    # NOTE(readability): This guards mixed MinIO/AWS deployments where endpoint
    # scheme should not be overwritten by normalization logic.
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://minio.kaiyuan-ray.svc.cluster.local:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "minio")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "minio123")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("LANCE_AWS_ALLOW_HTTP", "1")

    options = default_storage_options(None)
    assert options is not None
    assert options["aws_endpoint"] == "http://minio.kaiyuan-ray.svc.cluster.local:9000"
    assert options["aws_allow_http"] == "true"


def test_default_storage_options_adds_scheme_for_plain_endpoint(monkeypatch):
    """Ensure plain host:port endpoints receive scheme from allow-http toggle.

    Inputs/outputs:
        Seeds MINIO env vars and verifies synthesized endpoint URL.

    Side effects:
        Mutates process environment via pytest `monkeypatch`.

    Assumptions:
        `LANCE_AWS_ALLOW_HTTP=0` maps to HTTPS endpoint and `"false"` flag.
    """
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("MINIO_ENDPOINT", "minio.kaiyuan-ray.svc.cluster.local:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "minio")
    monkeypatch.setenv("MINIO_SECRET_KEY", "minio123")
    monkeypatch.setenv("LANCE_AWS_ALLOW_HTTP", "0")

    options = default_storage_options(None)
    assert options is not None
    assert options["aws_endpoint"] == "https://minio.kaiyuan-ray.svc.cluster.local:9000"
    assert options["aws_allow_http"] == "false"
