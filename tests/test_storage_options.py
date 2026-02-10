from __future__ import annotations

from datafiner.dataset_utils import default_storage_options


def test_default_storage_options_preserves_scheme_and_string_allow_http(monkeypatch):
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
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("MINIO_ENDPOINT", "minio.kaiyuan-ray.svc.cluster.local:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "minio")
    monkeypatch.setenv("MINIO_SECRET_KEY", "minio123")
    monkeypatch.setenv("LANCE_AWS_ALLOW_HTTP", "0")

    options = default_storage_options(None)
    assert options is not None
    assert options["aws_endpoint"] == "https://minio.kaiyuan-ray.svc.cluster.local:9000"
    assert options["aws_allow_http"] == "false"

