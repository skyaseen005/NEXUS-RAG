"""
rag.py — RAG orchestration engine
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Generator, List, Tuple

from backend.config import SYSTEM_TEMPLATE
from backend.vectorstore import VectorStore
from backend.llm import build_messages, quick_chat, stream_chat


class RAGEngine:
    def __init__(self):
        self.store   = VectorStore()
        self.history: List[Dict] = []

    # ── Context formatting ─────────────────────────────────────────────────────

    def _context(self, docs: List[Dict]) -> str:
        if not docs:
            return "(No relevant documents retrieved.)"
        parts = []
        for i, doc in enumerate(docs, 1):
            meta  = doc.get("metadata", {})
            src   = meta.get("source", "?")
            score = doc.get("score", 0.0)
            dtype = meta.get("doc_type", "")
            label = f"[Doc {i}] {src}" + (f" ({dtype})" if dtype else "")
            label += f"  ·  relevance {score:.0%}"
            parts.append(f"{label}\n{doc['text']}")
        return "\n\n─────────────────────\n\n".join(parts)

    # ── Query enhancement ──────────────────────────────────────────────────────

    def _enhance(self, query: str) -> str:
        try:
            result = quick_chat(
                f"Rewrite for better document retrieval. "
                f"Return ONLY the rewritten query, nothing else:\n{query}"
            ).strip().strip('"').strip("'")
            return result if 5 < len(result) < 300 else query
        except Exception:
            return query

    # ── Streaming answer ───────────────────────────────────────────────────────

    def stream_answer(
        self, query: str
    ) -> Generator[Tuple[str, object], None, None]:
        eq = self._enhance(query)
        yield ("enhanced_query", eq)

        docs = self.store.mmr_query(eq)
        yield ("docs", docs)

        context  = self._context(docs)
        sys_p    = SYSTEM_TEMPLATE.format(context=context)
        messages = build_messages(sys_p, self.history, query)

        full = ""
        for token in stream_chat(messages):
            full += token
            yield ("token", token)

        self.history.append({"role": "user",      "content": query})
        self.history.append({"role": "assistant",  "content": full})
        yield ("done", full)

    # ── Conversation management ────────────────────────────────────────────────

    def clear_history(self):
        self.history.clear()

    @property
    def turn_count(self) -> int:
        return len(self.history) // 2

    def export(self, path: str):
        lines = []
        for msg in self.history:
            role = "YOU" if msg["role"] == "user" else "NEXUS"
            lines.append(f"[{role}]\n{msg['content']}\n")
        Path(path).write_text("\n".join(lines), encoding="utf-8")
