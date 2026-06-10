"""Pytest fixtures for isolated test environments."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.config.settings import get_settings


@pytest.fixture(autouse=True)
def mock_ml_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock SentenceTransformer and CrossEncoder to avoid downloading models in tests."""
    mock_transformer = MagicMock()
    mock_transformer.encode.side_effect = lambda texts, **kwargs: (
        np.ones((len(texts), 1024), dtype=np.float32)
        if isinstance(texts, list)
        else np.ones(1024, dtype=np.float32)
    )

    mock_cross_encoder = MagicMock()
    mock_cross_encoder.predict.side_effect = lambda pairs, **kwargs: np.array(
        [1.0 / (i + 1) for i in range(len(pairs))], dtype=np.float32
    )

    monkeypatch.setattr(
        "app.embeddings.embedding_service.SentenceTransformer",
        lambda *args, **kwargs: mock_transformer,
    )
    monkeypatch.setattr(
        "app.reranker.reranker.CrossEncoder",
        lambda *args, **kwargs: mock_cross_encoder,
    )


@pytest.fixture(autouse=True)
def mock_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock Redis client to raise ConnectionError immediately and avoid network timeouts."""
    try:
        import redis
        mock_client = MagicMock()
        mock_client.ping.side_effect = redis.ConnectionError("Connection refused")
        monkeypatch.setattr(redis, "Redis", lambda *args, **kwargs: mock_client)
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def isolated_data_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use temporary directories for vector/BM25 stores during tests."""
    vector_dir = tmp_path / "vector_store"
    bm25_dir = tmp_path / "bm25_store"
    vector_dir.mkdir()
    bm25_dir.mkdir()

    settings = get_settings()
    monkeypatch.setattr(settings, "vector_store_dir", vector_dir)
    monkeypatch.setattr(settings, "bm25_store_dir", bm25_dir)

    yield

    get_settings.cache_clear()
