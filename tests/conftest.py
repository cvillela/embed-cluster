"""Shared pytest fixtures and marker registration."""

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: requires a CUDA GPU and RAPIDS / FAISS GPU runtime",
    )
