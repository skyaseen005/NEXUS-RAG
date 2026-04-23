"""
app.py — NEXUS RAG · FastAPI application entry point
All business logic lives in config / ingest / vectorstore / llm / rag.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import HOST, PORT, GROQ_MODEL, EMBED_MODEL, EXPORT_DIR, BASE_DIR
from backend.ingest import load_pdf, load_txt, load_url, load_raw_text
from backend.rag import RAGEngine

# ── App setup ──────────────────────────────────────────────────────────────────

app    = FastAPI(title="NEXUS RAG API", version="2.0")
engine = RAGEngine()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str

class UrlRequest(BaseModel):
    url: str

class TextRequest(BaseModel):
    text: str
    name: str = "paste"


# ── Stats ──────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats():
    sources = engine.store.list_sources()
    return {
        "chunk_count": engine.store.count(),
        "doc_count":   len(sources),
        "turn_count":  engine.turn_count,
        "model":       GROQ_MODEL,
        "embed_model": EMBED_MODEL,
    }


# ── Documents ──────────────────────────────────────────────────────────────────

@app.get("/api/documents")
async def list_documents():
    return {"documents": engine.store.list_sources()}


@app.delete("/api/documents/{source:path}")
async def delete_document(source: str):
    n = engine.store.delete_source(source)
    if n == 0:
        raise HTTPException(404, f"Source '{source}' not found")
    return {"deleted": n, "source": source}


# ── Ingestion ──────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, "Only PDF, TXT, and MD files are supported")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks = load_pdf(tmp_path) if suffix == ".pdf" else load_txt(tmp_path)
        for c in chunks:
            c["source"] = file.filename
        added = engine.store.add_chunks(chunks)
    finally:
        os.unlink(tmp_path)

    return {
        "added":   added,
        "total":   len(chunks),
        "source":  file.filename,
        "skipped": len(chunks) - added,
    }


@app.post("/api/add-url")
async def add_url(req: UrlRequest):
    try:
        chunks = load_url(req.url)
    except Exception as e:
        raise HTTPException(400, str(e))
    added = engine.store.add_chunks(chunks)
    return {"added": added, "total": len(chunks), "skipped": len(chunks) - added}


@app.post("/api/add-text")
async def add_text(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty")
    chunks = load_raw_text(req.text, name=req.name)
    added  = engine.store.add_chunks(chunks)
    return {"added": added, "total": len(chunks), "skipped": len(chunks) - added}


# ── Chat (streaming SSE) ───────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if engine.store.count() == 0:
        raise HTTPException(400, "No documents indexed. Add documents first.")

    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def worker():
        try:
            for event, data in engine.stream_answer(req.query):
                asyncio.run_coroutine_threadsafe(q.put((event, data)), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(q.put(("error", str(e))), loop)
        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

    threading.Thread(target=worker, daemon=True).start()

    async def generate():
        while True:
            item = await q.get()
            if item is None:
                break
            event, data = item

            if event == "token":
                payload = {"type": "token", "content": data}
            elif event == "enhanced_query":
                payload = {"type": "enhanced_query", "content": data}
            elif event == "docs":
                payload = {
                    "type": "docs",
                    "docs": [
                        {
                            "source":   d["metadata"].get("source", "?"),
                            "score":    d.get("score", 0.0),
                            "preview":  d["text"][:120].replace("\n", " "),
                            "doc_type": d["metadata"].get("doc_type", ""),
                        }
                        for d in data
                    ],
                }
            elif event == "done":
                payload = {"type": "done"}
            elif event == "error":
                payload = {"type": "error", "content": data}
            else:
                continue

            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ── History ────────────────────────────────────────────────────────────────────

@app.delete("/api/history")
async def clear_history():
    engine.clear_history()
    return {"ok": True}


@app.get("/api/export")
async def export_session():
    if not engine.history:
        raise HTTPException(400, "No conversation to export")
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = EXPORT_DIR / f"session_{ts}.txt"
    engine.export(str(path))
    return FileResponse(
        str(path),
        filename=f"nexus_session_{ts}.txt",
        media_type="text/plain",
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║   N E X U S   R A G   v2.0          ║")
    print(f"  ║   http://{HOST}:{PORT}               ║")
    print("  ╚══════════════════════════════════════╝")
    print()
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False, log_level="warning")
