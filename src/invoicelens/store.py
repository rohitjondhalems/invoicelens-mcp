"""
Persistent FAISS vector store, one index per collection.

Layout on disk (data_dir/collections/<name>/):
    index.faiss      IndexIDMap2(IndexFlatIP)  — exact cosine search on normalised vectors
    chunks.json      chunk id -> {doc_id, page, chunk_type, section, text}
    documents.json   doc_id -> {filename, pages, sha256, chunk_count, ingested_at}
    manifest.json    {embedder, dim} — guards against mixing embedding models
    pdfs/<doc_id>.pdf  original file, so extraction can run later by doc_id

IndexFlatIP is exact (no approximation) and fast to well beyond demo scale
(~100k chunks). Swap for IndexHNSWFlat or a managed store when it isn't.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

_COLLECTION_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


class StoreError(ValueError):
    pass


@dataclass
class StoredChunk:
    chunk_id: int
    doc_id: str
    page: int
    chunk_type: str
    section: str | None
    text: str


@dataclass
class SearchHit:
    score: float
    chunk: StoredChunk
    filename: str


def validate_collection(name: str) -> str:
    if not _COLLECTION_RE.match(name):
        raise StoreError(f"Invalid collection name {name!r}: use 1-63 lowercase letters, "
                         "digits, '-' or '_', starting with a letter or digit.")
    return name


class VectorStore:
    def __init__(self, root: Path, collection: str, embedder_name: str, dim: int):
        self.dir = Path(root) / "collections" / validate_collection(collection)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "pdfs").mkdir(exist_ok=True)
        self.collection = collection
        self.embedder_name, self.dim = embedder_name, dim
        self._lock = threading.RLock()
        self._load()

    # ── persistence ─────────────────────────────────────────────────────
    def _load(self) -> None:
        manifest = self.dir / "manifest.json"
        if manifest.exists():
            m = json.loads(manifest.read_text())
            if (m["embedder"], m["dim"]) != (self.embedder_name, self.dim):
                raise StoreError(
                    f"Collection '{self.collection}' was built with {m['embedder']} "
                    f"({m['dim']}-d) but the server is configured for {self.embedder_name} "
                    f"({self.dim}-d). Vectors from different models are not comparable: use a "
                    "new collection name, or delete this collection and re-ingest.")
        else:
            manifest.write_text(json.dumps({"embedder": self.embedder_name, "dim": self.dim}))

        idx_path = self.dir / "index.faiss"
        self.index = (faiss.read_index(str(idx_path)) if idx_path.exists()
                      else faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim)))
        self.chunks: dict[int, StoredChunk] = {
            int(k): StoredChunk(**v) for k, v in self._read_json("chunks.json").items()}
        self.documents: dict[str, dict] = self._read_json("documents.json")
        self._next_id = max(self.chunks, default=-1) + 1

    def _read_json(self, name: str) -> dict:
        p = self.dir / name
        return json.loads(p.read_text()) if p.exists() else {}

    def _write_json(self, name: str, obj: dict) -> None:
        tmp = self.dir / f".{name}.tmp"
        tmp.write_text(json.dumps(obj, ensure_ascii=False))
        tmp.replace(self.dir / name)          # atomic on POSIX and Windows

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.dir / ".index.tmp"))
        (self.dir / ".index.tmp").replace(self.dir / "index.faiss")
        self._write_json("chunks.json", {str(k): asdict(v) for k, v in self.chunks.items()})
        self._write_json("documents.json", self.documents)

    # ── writes ──────────────────────────────────────────────────────────
    def add_document(self, doc_id: str, doc_meta: dict, pdf_bytes: bytes,
                     chunks: list[dict], vectors: np.ndarray) -> bool:
        """Insert a document's chunks. Re-ingesting the same doc_id replaces it.
        Returns True if an existing version was replaced."""
        if len(chunks) != len(vectors):
            raise StoreError("chunks and vectors length mismatch")
        if vectors.shape[1] != self.dim:
            raise StoreError(f"vector dim {vectors.shape[1]} != index dim {self.dim}")
        with self._lock:
            replaced = self._remove(doc_id)
            ids = np.arange(self._next_id, self._next_id + len(chunks), dtype="int64")
            self._next_id += len(chunks)
            if len(ids):
                self.index.add_with_ids(np.ascontiguousarray(vectors, dtype="float32"), ids)
            for cid, c in zip(ids, chunks):
                self.chunks[int(cid)] = StoredChunk(chunk_id=int(cid), doc_id=doc_id, **c)
            self.documents[doc_id] = {**doc_meta, "chunk_count": len(chunks)}
            (self.dir / "pdfs" / f"{doc_id}.pdf").write_bytes(pdf_bytes)
            self._save()
            return replaced

    def _remove(self, doc_id: str) -> bool:
        ids = [cid for cid, c in self.chunks.items() if c.doc_id == doc_id]
        if ids:
            self.index.remove_ids(np.asarray(ids, dtype="int64"))
            for cid in ids:
                del self.chunks[cid]
        return self.documents.pop(doc_id, None) is not None

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            removed = self._remove(doc_id)
            (self.dir / "pdfs" / f"{doc_id}.pdf").unlink(missing_ok=True)
            if removed:
                self._save()
            return removed

    # ── reads ───────────────────────────────────────────────────────────
    def search(self, qvec: np.ndarray, k: int = 5, doc_id: str | None = None,
               chunk_types: set[str] | None = None) -> list[SearchHit]:
        with self._lock:
            if self.index.ntotal == 0:
                return []
            filtered = doc_id is not None or chunk_types
            # Over-fetch when filtering, since FAISS filters after scoring here
            fetch = self.index.ntotal if filtered else min(k, self.index.ntotal)
            scores, ids = self.index.search(np.ascontiguousarray(qvec.reshape(1, -1)), fetch)
            hits = []
            for score, cid in zip(scores[0], ids[0]):
                if cid < 0:
                    continue
                c = self.chunks[int(cid)]
                if doc_id is not None and c.doc_id != doc_id:
                    continue
                if chunk_types and c.chunk_type not in chunk_types:
                    continue
                filename = self.documents.get(c.doc_id, {}).get("filename", "")
                hits.append(SearchHit(float(score), c, filename))
                if len(hits) >= k:
                    break
            return hits

    def get_pdf(self, doc_id: str) -> bytes:
        p = self.dir / "pdfs" / f"{doc_id}.pdf"
        if doc_id not in self.documents or not p.exists():
            raise StoreError(f"No document '{doc_id}' in collection '{self.collection}'. "
                             "Use invoicelens_search_documents to find doc_ids, or ingest the PDF first.")
        return p.read_bytes()

    @property
    def size(self) -> int:
        return int(self.index.ntotal)
