"""
llm.py — Groq LLM interface (streaming + one-shot)
"""
from __future__ import annotations

from typing import Dict, Generator, List

from backend.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMP, GROQ_MAX_TOKENS


def _client():
    from groq import Groq
    return Groq(api_key=GROQ_API_KEY)


def build_messages(
    system:    str,
    history:   List[Dict],
    query:     str,
    max_turns: int = 8,
) -> List[Dict]:
    msgs = [{"role": "system", "content": system}]
    msgs.extend(history[-(max_turns * 2):])
    msgs.append({"role": "user", "content": query})
    return msgs


def stream_chat(messages: List[Dict]) -> Generator[str, None, None]:
    stream = _client().chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=GROQ_TEMP,
        max_tokens=GROQ_MAX_TOKENS,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


def quick_chat(prompt: str) -> str:
    resp = _client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=GROQ_TEMP,
        max_tokens=256,
        stream=False,
    )
    return resp.choices[0].message.content
