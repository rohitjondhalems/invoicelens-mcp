import numpy as np
import pytest

from invoicelens.embeddings import HashingEmbedder
from invoicelens.service import InvoiceLensService
from invoicelens.store import StoreError, VectorStore

from .conftest import ACME, INITECH, NORTHWIND, sample


def test_reingest_replaces_instead_of_duplicating(service):
    first = service.ingest(sample(ACME), ACME)
    size = service.store().size
    again = service.ingest(sample(ACME), ACME)
    assert first.doc_id == again.doc_id
    assert again.replaced_existing and not first.replaced_existing
    assert service.store().size == size


def test_index_persists_across_restarts(settings, service):
    service.ingest(sample(NORTHWIND), NORTHWIND)
    restarted = InvoiceLensService(settings)
    hits = restarted.search("late payment penalty", k=1).results
    assert hits[0].filename == NORTHWIND and hits[0].page == 2


def test_search_filters(loaded):
    initech = loaded.search("Initech", k=1).results[0].doc_id
    only = loaded.search("total", k=10, doc_id=initech).results
    assert only and {h.doc_id for h in only} == {initech}
    totals = loaded.search("amount", k=10, chunk_types=["totals"]).results
    assert len(totals) == 4 and {h.chunk_type for h in totals} == {"totals"}


def test_collections_are_isolated(service):
    service.ingest(sample(ACME), ACME, collection="client-a")
    service.ingest(sample(INITECH), INITECH, collection="client-b")
    assert {h.filename for h in service.search("invoice", k=10, collection="client-a").results} == {ACME}
    assert {h.filename for h in service.search("invoice", k=10, collection="client-b").results} == {INITECH}


def test_refuses_to_mix_embedding_models(tmp_path):
    VectorStore(tmp_path, "c", "model-a", 8)
    with pytest.raises(StoreError, match="Vectors from different models are not comparable"):
        VectorStore(tmp_path, "c", "model-b", 16)


def test_rejects_bad_collection_names(tmp_path):
    with pytest.raises(StoreError, match="Invalid collection name"):
        VectorStore(tmp_path, "../etc", "m", 8)


def test_delete_document(tmp_path):
    emb = HashingEmbedder(64)
    s = VectorStore(tmp_path, "c", emb.name, emb.dim)
    chunk = {"page": 1, "chunk_type": "text", "section": None, "text": "hello world"}
    s.add_document("doc_a", {"filename": "a.pdf"}, b"%PDF", [chunk], emb.embed(["hello world"]))
    assert s.size == 1 and s.delete_document("doc_a") and s.size == 0
    assert s.search(emb.embed(["hello"])[0]) == []


def test_hashing_embedder_is_normalised_and_deterministic():
    e = HashingEmbedder(256)
    a, b = e.embed(["late payment penalty"]), e.embed(["late payment penalty"])
    assert np.allclose(a, b) and np.isclose(np.linalg.norm(a), 1.0)
    q, near, far = e.embed(["late payment penalty", "penalty for late payment", "pressure gauge"])
    assert q @ near > q @ far
