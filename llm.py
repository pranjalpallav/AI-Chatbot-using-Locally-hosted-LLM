"""Prompt building + streaming from a free hosted LLM API (OpenAI-compatible)."""
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from utils import strip_think

load_dotenv()

def _cfg(name: str, default: str = "") -> str:
    """Read a setting from env/.env (local) or Streamlit secrets (Streamlit Cloud)."""
    val = os.getenv(name)
    if val:
        return val
    try:
        import streamlit as st
        return str(st.secrets.get(name, default))
    except Exception:
        return default


# Defaults to Groq's free tier. Change these in .env / Streamlit secrets to use another provider.
API_KEY = _cfg("LLM_API_KEY")
BASE_URL = _cfg("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = _cfg("LLM_MODEL", "qwen/qwen3.8-27b")
TEMPERATURE = 0.3


def _today() -> str:
    return time.strftime("%A, %d %B %Y")


def chat_system() -> str:
    return ("You are a helpful, friendly AI assistant. Be clear and concise. "
            f"Today's date is {_today()}. You have NO internet access in this mode, so your "
            "knowledge may be out of date. Never provide URLs or links, because you cannot "
            "verify them; suggest the user turn on web search for current information.")


def rag_system() -> str:
    return ("You are a helpful assistant. Answer using ONLY the numbered context passages "
            "provided. They come from the user's documents and/or web pages fetched just now. "
            "Cite the passages you used like [1] or [2]. Never write URLs and never invent "
            "sources; the app displays the real links. If the context does not contain the "
            f"answer, say so. Today's date is {_today()}. Be clear and concise.")


def _label(h: dict) -> str:
    if h.get("kind") == "web":
        return f"web page: {h['source']}, fetched {h.get('fetched', '')}"
    return f"{h['source']}, page {h['page']}"


def build_messages(question: str, hits: list[dict], history: list[dict]) -> list[dict]:
    history = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]
    if not hits:   # plain chat
        return [{"role": "system", "content": chat_system()}, *history,
                {"role": "user", "content": question}]
    context = "\n\n".join(f"[{i}] ({_label(h)})\n{h['text']}" for i, h in enumerate(hits, start=1))
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return [{"role": "system", "content": rag_system()}, *history, {"role": "user", "content": user}]


def _raw_stream(messages: list[dict]):
    if not API_KEY:
        yield ("⚠️ No API key found. Set `LLM_API_KEY` in a `.env` file (local) or in Streamlit "
               "Secrets (cloud), then restart the app.")
        return
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    try:
        stream = client.chat.completions.create(
            model=MODEL, messages=messages, stream=True, temperature=TEMPERATURE)
        for chunk in stream:
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if piece:
                yield piece
    except Exception as e:      # rate limit, bad key, network, wrong model name...
        yield f"\n\n⚠️ LLM API error: {str(e)[:300]}"


def stream_answer(messages: list[dict]):
    """Yield only the visible answer text (any <think> blocks are stripped)."""
    yield from strip_think(_raw_stream(messages))
