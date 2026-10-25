"""
Embedding providers. All return L2-normalised float32 vectors, so FAISS inner
product == cosine similarity.

  AzureOpenAIEmbedder / OpenAIEmbedder : text-embedding-3-large (3072-d) by default
  HashingEmbedder                      : offline, deterministic, no network.

HashingEmbedder is a lexical (bag of word n-grams) model, NOT semantic: it finds
"late payment penalty" but not "fees for paying after the deadline". It exists so
tests and CI run without keys and so the server never fails to start. Use a real
model for anything you'd show as retrieval quality.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from invoicelens.config import Settings


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype="float32")
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


class HashingEmbedder:
    """Feature-hashed unigrams + bigrams, signed, log-scaled. Deterministic."""

    _WORD = re.compile(r"[a-z0-9]+(?:[.,/-][a-z0-9]+)*")

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"local-hashing-{dim}"

    def _features(self, text: str) -> list[str]:
        words = self._WORD.findall(text.lower())
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for f in self._features(t):
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        out = np.sign(out) * np.log1p(np.abs(out))
        return _normalise(out)


class _OpenAICompatibleEmbedder:
    """Shared batching for OpenAI and Azure OpenAI (same embeddings API)."""

    batch_size = 64

    def __init__(self, client, model: str, dim: int, name: str):
        self._client, self._model, self.dim, self.name = client, model, dim, name

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[i:i + self.batch_size]]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        arr = np.asarray(vectors, dtype="float32")
        if arr.shape[1] != self.dim:
            raise ValueError(f"{self.name} returned {arr.shape[1]}-d vectors, expected {self.dim}")
        return _normalise(arr)


# Native output sizes of the OpenAI embedding family
_DIMS = {"text-embedding-3-large": 3072, "text-embedding-3-small": 1536, "text-embedding-ada-002": 1536}


def _dim_for(model: str) -> int:
    for key, dim in _DIMS.items():
        if key in model:
            return dim
    return 3072


def make_embedder(settings: Settings) -> Embedder:
    provider = settings.resolve(settings.embedding_provider)
    if provider == "azure":
        from openai import AzureOpenAI

        client = AzureOpenAI(azure_endpoint=settings.azure_openai_endpoint,
                             api_key=settings.azure_openai_api_key,
                             api_version=settings.azure_openai_api_version, max_retries=3)
        model = settings.azure_embedding_deployment
        return _OpenAICompatibleEmbedder(client, model, _dim_for(model), f"azure:{model}")
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key, max_retries=3)
        model = settings.openai_embedding_model
        return _OpenAICompatibleEmbedder(client, model, _dim_for(model), f"openai:{model}")
    return HashingEmbedder()
