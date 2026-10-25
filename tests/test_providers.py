"""Azure/OpenAI code paths, against fake clients shaped like the OpenAI SDK (no network)."""
from types import SimpleNamespace

import numpy as np
import pytest

from invoicelens.config import Settings
from invoicelens.embeddings import HashingEmbedder, _OpenAICompatibleEmbedder, make_embedder
from invoicelens.extraction import FallbackExtractor, LLMExtractor, RuleExtractor, make_extractor
from invoicelens.models import Invoice
from invoicelens.parsing import parse_pdf

from .conftest import ACME, sample


class FakeEmbeddings:
    def __init__(self, dim):
        self.dim, self.calls = dim, []

    def create(self, model, input):
        self.calls.append(len(input))
        data = [SimpleNamespace(index=i, embedding=[float(i + 1)] * self.dim) for i in range(len(input))]
        return SimpleNamespace(data=list(reversed(data)))       # out of order on purpose


def test_openai_embedder_batches_orders_and_normalises():
    client = SimpleNamespace(embeddings=FakeEmbeddings(3072))
    emb = _OpenAICompatibleEmbedder(client, "text-embedding-3-large", 3072, "azure:text-embedding-3-large")
    out = emb.embed([f"t{i}" for i in range(150)])
    assert client.embeddings.calls == [64, 64, 22]
    assert out.shape == (150, 3072) and np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_embedder_rejects_wrong_dimension():
    client = SimpleNamespace(embeddings=FakeEmbeddings(1536))
    emb = _OpenAICompatibleEmbedder(client, "text-embedding-3-large", 3072, "x")
    with pytest.raises(ValueError, match="1536-d vectors, expected 3072"):
        emb.embed(["a"])


class FakeChat:
    def __init__(self, parsed):
        self.parsed, self.messages = parsed, None

    def parse(self, model, temperature, response_format, messages):
        assert response_format is Invoice and temperature == 0
        self.messages = messages
        msg = SimpleNamespace(parsed=self.parsed, refusal=None if self.parsed else "no")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_llm_extractor_sends_document_and_feedback():
    chat = FakeChat(Invoice(vendor_name="Acme", total=1.0))
    client = SimpleNamespace(chat=SimpleNamespace(completions=chat))
    ex = LLMExtractor(client, "gpt-4o", "azure:gpt-4o")
    inv = ex.extract(parse_pdf(sample(ACME)), feedback="Line items sum to X but subtotal is Y")
    user = chat.messages[1]["content"]
    assert inv.vendor_name == "Acme"
    assert "| Description | Qty | Unit Price | Amount |" in user       # tables sent as markdown
    assert "A previous extraction failed these checks" in user and "subtotal is Y" in user
    assert "Do NOT recompute" in chat.messages[0]["content"]


def test_llm_refusal_falls_back_to_rules():
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeChat(None)))
    ex = FallbackExtractor(LLMExtractor(client, "gpt-4o", "azure:gpt-4o"), RuleExtractor())
    assert ex.extract(parse_pdf(sample(ACME))).total == 3369.49
    assert ex.name.startswith("rules (fallback")


def test_provider_resolution(tmp_path):
    local = Settings(data_dir=tmp_path, _env_file=None)
    assert isinstance(make_embedder(local), HashingEmbedder)
    assert isinstance(make_extractor(local), RuleExtractor)
    azure = Settings(data_dir=tmp_path, azure_openai_endpoint="https://x.openai.azure.com",
                     azure_openai_api_key="k", _env_file=None)
    emb = make_embedder(azure)
    assert (emb.name, emb.dim) == ("azure:text-embedding-3-large", 3072)
    assert make_extractor(azure).name == "azure:gpt-4o"
