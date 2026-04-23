"""
config.py — Centralised configuration loaded from environment / .env
"""
from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env from the project root (two levels up from backend/)
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")


def _get(key: str, default: str | None = None, required: bool = False) -> str:
    val = os.getenv(key, default)
    if required and not val:
        raise RuntimeError(
            f"Missing required environment variable: {key}\n"
            f"Copy .env.example → .env and fill in the value."
        )
    return val  # type: ignore[return-value]


# ── Groq / LLM ────────────────────────────────────────────────────────────────
GROQ_API_KEY: str   = _get("GROQ_API_KEY", required=True)
GROQ_MODEL: str     = _get("GROQ_MODEL",   "llama-3.3-70b-versatile")
GROQ_TEMP: float    = float(_get("GROQ_TEMP",       "0.15"))
GROQ_MAX_TOKENS: int = int(_get("GROQ_MAX_TOKENS",  "2048"))

# ── Embedding ──────────────────────────────────────────────────────────────────
EMBED_MODEL: str = _get("EMBED_MODEL", "all-MiniLM-L6-v2")

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE: int    = int(_get("CHUNK_SIZE",    "550"))
CHUNK_OVERLAP: int = int(_get("CHUNK_OVERLAP", "80"))

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K: int        = int(_get("TOP_K",        "5"))
FETCH_K: int      = int(_get("FETCH_K",      "14"))
MMR_LAMBDA: float = float(_get("MMR_LAMBDA", "0.65"))

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = _ROOT
DATA_DIR   = BASE_DIR / "nexus_data"
CHROMA_DIR = DATA_DIR / "chroma"
EXPORT_DIR = DATA_DIR / "exports"

for _d in (DATA_DIR, CHROMA_DIR, EXPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Server ─────────────────────────────────────────────────────────────────────
HOST: str = _get("HOST", "0.0.0.0")
PORT: int = int(_get("PORT", "8000"))

# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_TEMPLATE = """\
You are **NEXUS**, a precision research AI powered by a live knowledge base.

Rules:
1. Answer using the retrieved documents below.
2. Cite claims with [Doc N] — e.g., "The policy changed in 2021 [Doc 2]."
3. If multiple docs support a claim, cite all.
4. If docs are insufficient say so honestly, then share what you know.
5. Use markdown: headers, bold, lists for clarity.
6. Never fabricate facts.

────────────────────────────────────────
RETRIEVED DOCUMENTS
────────────────────────────────────────
{context}
────────────────────────────────────────
"""
