# NEXUS RAG v2.0

A production-ready Retrieval-Augmented Generation app — FastAPI backend with a polished React frontend.

## Features

- **MMR retrieval** — Maximal Marginal Relevance for diverse, relevant chunks  
- **Query enhancement** — LLM rewrites your question for better recall  
- **Streaming answers** — Token-by-token SSE streaming  
- **Multiple sources** — PDF, TXT/MD, URLs, raw paste  
- **Persistent vector store** — ChromaDB on disk  
- **Citation-aware** — NEXUS cites `[Doc N]` in every answer  

## Quick Start

```bash
# 1. Clone / unzip the project
# 2. Copy and configure environment
cp .env.example .env
# Edit .env — set GROQ_API_KEY

# 3. Launch (creates venv, installs deps, starts server)
./start.sh
# → Open http://localhost:8000
```

Get a free Groq API key at https://console.groq.com

## Project Structure

```
nexus-rag/
├── .env.example          ← copy → .env and fill in secrets
├── .env                  ← NOT committed (in .gitignore)
├── .gitignore
├── start.sh              ← one-command launcher
├── README.md
│
├── backend/
│   ├── app.py            ← FastAPI routes (entry point)
│   ├── config.py         ← all settings, loaded from .env
│   ├── ingest.py         ← PDF / TXT / URL / raw-text loaders + chunking
│   ├── vectorstore.py    ← ChromaDB wrapper + MMR retrieval
│   ├── llm.py            ← Groq streaming + one-shot helpers
│   ├── rag.py            ← RAG engine (orchestration)
│   └── requirements.txt
│
├── frontend/
│   └── index.html        ← single-file React app
│
└── nexus_data/           ← auto-created, gitignored
    ├── chroma/           ← vector store
    └── exports/          ← session export files
```

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | **required** | Your Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM model |
| `GROQ_TEMP` | `0.15` | Temperature |
| `GROQ_MAX_TOKENS` | `2048` | Max response tokens |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `CHUNK_SIZE` | `550` | Characters per chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between chunks |
| `TOP_K` | `5` | Chunks returned per query |
| `FETCH_K` | `14` | Candidates before MMR |
| `MMR_LAMBDA` | `0.65` | MMR diversity weight |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
