"""
ingest.py — Document loading and chunking
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Dict, List

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP


# ── Text cleaning ──────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str) -> List[Dict]:
    text      = _clean(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: List[Dict] = []
    current: List[str] = []
    current_len        = 0
    chunk_idx          = 0

    def _make(body: str, idx: int) -> Dict:
        cid = hashlib.md5(f"{source}|{idx}|{body[:60]}".encode()).hexdigest()[:14]
        return {
            "id":          cid,
            "text":        body,
            "source":      source,
            "chunk_index": idx,
            "char_count":  len(body),
            "timestamp":   int(time.time()),
        }

    for sent in sentences:
        if current_len + len(sent) > CHUNK_SIZE and current:
            chunks.append(_make(" ".join(current), chunk_idx))
            chunk_idx += 1
            tail, tail_len = [], 0
            for s in reversed(current):
                if tail_len + len(s) > CHUNK_OVERLAP:
                    break
                tail.insert(0, s)
                tail_len += len(s)
            current, current_len = tail, tail_len
        current.append(sent)
        current_len += len(sent) + 1

    if current:
        chunks.append(_make(" ".join(current), chunk_idx))
    return chunks


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_pdf(path: str) -> List[Dict]:
    from pypdf import PdfReader
    reader    = PdfReader(path)
    full_text = "\n\n".join(
        p.extract_text() for p in reader.pages if p.extract_text()
    )
    chunks = chunk_text(full_text, Path(path).name)
    for c in chunks:
        c["doc_type"] = "pdf"
        c["pages"]    = len(reader.pages)
    return chunks


def load_txt(path: str) -> List[Dict]:
    text   = Path(path).read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(text, Path(path).name)
    for c in chunks:
        c["doc_type"] = "txt"
    return chunks


def load_url(url: str) -> List[Dict]:
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urlparse

    resp = requests.get(
        url, timeout=20,
        headers={"User-Agent": "Mozilla/5.0 (NEXUS-RAG)"}
    )
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    main = soup.find("article") or soup.find("main") or soup.body
    text = main.get_text(separator=" ", strip=True) if main else soup.get_text()
    text = re.sub(r'\s{3,}', '\n', text)

    parsed = urlparse(url)
    source = parsed.netloc + (parsed.path[:40] if parsed.path != "/" else "")
    chunks = chunk_text(text, source)
    for c in chunks:
        c["doc_type"] = "url"
        c["url"]      = url
    return chunks


def load_raw_text(text: str, name: str = "paste") -> List[Dict]:
    chunks = chunk_text(text, name)
    for c in chunks:
        c["doc_type"] = "raw"
    return chunks


# ── Entry point ───────────────────────────────────────────────────────────────

def ingest(source: str) -> List[Dict]:
    s = source.strip()
    if s.startswith("http://") or s.startswith("https://"):
        return load_url(s)
    p = Path(s)
    if p.is_file():
        return load_pdf(s) if p.suffix.lower() == ".pdf" else load_txt(s)
    return load_raw_text(s)
