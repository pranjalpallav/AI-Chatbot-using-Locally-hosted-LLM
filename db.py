"""SQLite storage: chats, messages and per-chat document chunks (with embeddings)."""
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager

import numpy as np

DB_PATH = os.getenv("CHAT_DB", "chat.db")


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init() -> None:
    with conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, created REAL, updated REAL);
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            role TEXT NOT NULL, content TEXT NOT NULL, sources TEXT, created REAL);
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            source TEXT, page INTEGER, text TEXT, embedding BLOB);
        CREATE INDEX IF NOT EXISTS idx_msg_chat ON messages(chat_id);
        CREATE INDEX IF NOT EXISTS idx_chunk_chat ON chunks(chat_id);
        """)


def create_chat(title: str = "New chat") -> str:
    cid, now = uuid.uuid4().hex, time.time()
    with conn() as c:
        c.execute("INSERT INTO chats VALUES (?,?,?,?)", (cid, title, now, now))
    return cid


def list_chats(limit: int = 40) -> list[dict]:
    with conn() as c:
        rows = c.execute("SELECT * FROM chats ORDER BY updated DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_chat(chat_id: str):
    with conn() as c:
        r = c.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
    return dict(r) if r else None


def rename_chat(chat_id: str, title: str) -> None:
    with conn() as c:
        c.execute("UPDATE chats SET title=? WHERE id=?", (title, chat_id))


def delete_chat(chat_id: str) -> None:
    with conn() as c:
        c.execute("DELETE FROM chats WHERE id=?", (chat_id,))


def add_message(chat_id: str, role: str, content: str, sources: list | None = None) -> None:
    now = time.time()
    with conn() as c:
        c.execute("INSERT INTO messages (chat_id, role, content, sources, created) VALUES (?,?,?,?,?)",
                  (chat_id, role, content, json.dumps(sources) if sources else None, now))
        c.execute("UPDATE chats SET updated=? WHERE id=?", (now, chat_id))


def get_messages(chat_id: str) -> list[dict]:
    with conn() as c:
        rows = c.execute("SELECT role, content, sources FROM messages WHERE chat_id=? ORDER BY id",
                         (chat_id,)).fetchall()
    return [{"role": r["role"], "content": r["content"],
             "sources": json.loads(r["sources"]) if r["sources"] else []} for r in rows]


def add_chunks(chat_id: str, chunks: list[dict], vecs: np.ndarray) -> None:
    with conn() as c:
        c.executemany(
            "INSERT INTO chunks (chat_id, source, page, text, embedding) VALUES (?,?,?,?,?)",
            [(chat_id, ch["source"], ch["page"], ch["text"], vecs[i].astype("float32").tobytes())
             for i, ch in enumerate(chunks)])


def get_chunks(chat_id: str) -> tuple[list[dict], np.ndarray | None]:
    with conn() as c:
        rows = c.execute("SELECT source, page, text, embedding FROM chunks WHERE chat_id=? ORDER BY id",
                         (chat_id,)).fetchall()
    if not rows:
        return [], None
    chunks = [{"source": r["source"], "page": r["page"], "text": r["text"]} for r in rows]
    vecs = np.vstack([np.frombuffer(r["embedding"], dtype="float32") for r in rows])
    return chunks, vecs


def list_docs(chat_id: str) -> list[dict]:
    with conn() as c:
        rows = c.execute("SELECT source, COUNT(*) AS chunks FROM chunks WHERE chat_id=? GROUP BY source",
                         (chat_id,)).fetchall()
    return [dict(r) for r in rows]
