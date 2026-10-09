"""Prompt building + streaming from a local Ollama model (thinking hidden)."""
import os
import time

import ollama

from utils import strip_think

MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OPTIONS = {"num_ctx": 6144, "temperature": 0.3}


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
    history = [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    if not hits:   # plain chat
        return [{"role": "system", "content": chat_system()}, *history,
                {"role": "user", "content": question}]
    context = "\n\n".join(f"[{i}] ({_label(h)})\n{h['text']}" for i, h in enumerate(hits, start=1))
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return [{"role": "system", "content": rag_system()}, *history, {"role": "user", "content": user}]


def _raw_stream(messages: list[dict]):
    client = ollama.Client()
    try:
        stream = client.chat(model=MODEL, messages=messages, stream=True,
                             options=OPTIONS, think=False)
    except TypeError:                      # old ollama python package without `think`
        messages = [*messages[:-1],
                    {**messages[-1], "content": messages[-1]["content"] + " /no_think"}]
        stream = client.chat(model=MODEL, messages=messages, stream=True, options=OPTIONS)
    except ollama.ResponseError:           # model doesn't support the think flag
        stream = client.chat(model=MODEL, messages=messages, stream=True, options=OPTIONS)
    for chunk in stream:
        piece = chunk["message"]["content"]
        if piece:
            yield piece


def stream_answer(messages: list[dict]):
    """Yield only the visible answer text (any <think> blocks are stripped)."""
    yield from strip_think(_raw_stream(messages))
