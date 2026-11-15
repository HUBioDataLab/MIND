"""The cluster metadata download must fail loudly and leave no partial file behind."""

import importlib.util
from pathlib import Path

import pytest
import requests

SPEC = importlib.util.spec_from_file_location(
    "download_protein_clusters", Path(__file__).resolve().parents[1] / "data" / "download_protein_clusters.py")
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


def test_network_error_propagates_and_leaves_no_file(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise requests.exceptions.ConnectionError("no network")

    monkeypatch.setattr(download.requests, "get", fail)
    target = tmp_path / "clusters.tsv.gz"
    with pytest.raises(requests.exceptions.ConnectionError):
        download.download_file("http://example.invalid/x", target)
    assert not target.exists()


def test_http_error_propagates(tmp_path, monkeypatch):
    class Response:
        headers = {}

        def raise_for_status(self):
            raise requests.exceptions.HTTPError("404")

    monkeypatch.setattr(download.requests, "get", lambda *a, **k: Response())
    with pytest.raises(requests.exceptions.HTTPError):
        download.download_file("http://example.invalid/x", tmp_path / "clusters.tsv.gz")


def test_interrupted_stream_removes_the_partial_file(tmp_path, monkeypatch):
    class Response:
        headers = {"content-length": "100"}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"abc"
            raise requests.exceptions.ChunkedEncodingError("connection broken")

    monkeypatch.setattr(download.requests, "get", lambda *a, **k: Response())
    target = tmp_path / "clusters.tsv.gz"
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        download.download_file("http://example.invalid/x", target)
    assert not target.exists()
