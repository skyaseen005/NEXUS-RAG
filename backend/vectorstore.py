"""
vectorstore.py — ChromaDB vector store with MMR retrieval
"""
from __future__ import annotations

from typing import Dict, List, Optional

from backend.config import (
    CHROMA_DIR, EMBED_MODEL,
    TOP_K, FETCH_K, MMR_LAMBDA,
)


class VectorStore:
    def __init__(self):
        import chromadb
        self.client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            "nexus_knowledge", metadata={"hnsw:space": "cosine"}
        )
        self._model = None

    # ── Embedding ──────────────────────────────────────────────────────────────

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMBED_MODEL)
        return self._model

    def embed(self, texts: List[str]):
        return self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )

    # ── Write ──────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Dict]) -> int:
        if not chunks:
            return 0

        ids   = [c["id"]   for c in chunks]
        txts  = [c["text"] for c in chunks]
        metas = [{k: str(v) for k, v in c.items() if k != "text"} for c in chunks]

        try:
            existing = set(self.collection.get(ids=ids)["ids"])
        except Exception:
            existing = set()

        new_idx = [i for i, id_ in enumerate(ids) if id_ not in existing]
        if not new_idx:
            return 0

        embs = self.embed([txts[i] for i in new_idx]).tolist()
        self.collection.add(
            ids       =[ids[i]   for i in new_idx],
            documents =[txts[i]  for i in new_idx],
            embeddings=embs,
            metadatas =[metas[i] for i in new_idx],
        )
        return len(new_idx)

    # ── Query ──────────────────────────────────────────────────────────────────

    def _raw_query(self, query: str, k: int) -> List[Dict]:
        total = self.collection.count()
        if total == 0:
            return []
        k   = min(k, total)
        q   = self.embed([query])[0].tolist()
        res = self.collection.query(
            query_embeddings=[q], n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "id":       res["ids"][0][i],
                "text":     res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "score":    round(1.0 - float(res["distances"][0][i]), 4),
            }
            for i in range(len(res["ids"][0]))
        ]

    def mmr_query(
        self,
        query:   str,
        top_k:   int   = TOP_K,
        fetch_k: int   = FETCH_K,
        lam:     float = MMR_LAMBDA,
    ) -> List[Dict]:
        import numpy as np

        cands = self._raw_query(query, k=fetch_k)
        if not cands or len(cands) <= top_k:
            return cands

        q_emb    = self.embed([query])[0]
        doc_embs = self.embed([d["text"] for d in cands])
        selected, remaining = [], list(range(len(cands)))

        while len(selected) < top_k and remaining:
            scores = []
            for idx in remaining:
                rel = float(np.dot(q_emb, doc_embs[idx]))
                sim = (
                    max(float(np.dot(doc_embs[idx], doc_embs[s])) for s in selected)
                    if selected else 0.0
                )
                scores.append((idx, lam * rel - (1 - lam) * sim))
            best = max(scores, key=lambda x: x[1])[0]
            selected.append(best)
            remaining.remove(best)

        return [cands[i] for i in selected]

    # ── Metadata ───────────────────────────────────────────────────────────────

    def list_sources(self) -> List[Dict]:
        if self.collection.count() == 0:
            return []
        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        stats: Dict[str, Dict] = {}
        for m in all_meta:
            src = m.get("source", "unknown")
            if src not in stats:
                stats[src] = {
                    "source":   src,
                    "chunks":   0,
                    "doc_type": m.get("doc_type", "?"),
                }
            stats[src]["chunks"] += 1
        return sorted(stats.values(), key=lambda x: x["source"])

    def delete_source(self, source: str) -> int:
        res = self.collection.get(where={"source": source}, include=["metadatas"])
        ids = res["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return self.collection.count()
