#!/usr/bin/env bash
# ─── NEXUS RAG — Launcher ────────────────────────────────────────────────────
set -e

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║   N E X U S   R A G   v2.0             ║"
echo "  ║   FastAPI  ·  ChromaDB  ·  Groq         ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

# ── Checks ────────────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "  ERROR: Python 3 not found. Please install Python 3.9+"
  exit 1
fi

echo "  Python : $(python3 --version)"

# ── .env guard ────────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
  echo ""
  echo "  ⚠  .env not found — creating from .env.example"
  cp .env.example .env
  echo "  ✎  Edit .env and add your GROQ_API_KEY, then re-run this script."
  echo ""
  exit 1
fi

if grep -q "your_groq_api_key_here" .env 2>/dev/null; then
  echo ""
  echo "  ⚠  GROQ_API_KEY is not set in .env"
  echo "  ✎  Edit .env and replace 'your_groq_api_key_here' with your real key."
  echo "     Get a free key at https://console.groq.com"
  echo ""
  exit 1
fi

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "  Creating virtual environment…"
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --quiet --upgrade pip
echo "  Installing dependencies…"
pip install --quiet -r backend/requirements.txt

echo ""
echo "  ✔  Setup complete!"
echo "  ✔  Open http://localhost:8000 in your browser"
echo ""

cd backend
python app.py
