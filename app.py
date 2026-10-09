"""Streamlit assistant UI: chat history, PDF attachments in the message bar, live web search."""
from urllib.parse import urlparse

import streamlit as st

import db
from ingest import load_pdf, chunk_pages
from llm import MODEL, build_messages, stream_answer
from retriever import VectorStore, embed
from web import crawl

st.set_page_config(page_title="DocChat", page_icon="💬", layout="wide")
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {max-width: 820px; padding-top: 2rem;}
[data-testid="stSidebar"] .stButton > button {justify-content: flex-start; text-align: left;}
[data-testid="stChatInput"] {border-radius: 16px;}
</style>
""", unsafe_allow_html=True)

db.init()
ss = st.session_state
ss.setdefault("chat_id", None)      # None = a fresh chat that isn't saved yet
ss.setdefault("stores", {})         # chat_id -> VectorStore cache


# ---------- helpers ----------
def new_chat():
    ss.chat_id = None


def select_chat(cid: str):
    ss.chat_id = cid


def delete_chat(cid: str):
    db.delete_chat(cid)
    ss.stores.pop(cid, None)
    if ss.chat_id == cid:
        ss.chat_id = None


def ensure_chat() -> str:
    if ss.chat_id is None or db.get_chat(ss.chat_id) is None:
        ss.chat_id = db.create_chat()
    return ss.chat_id


def get_store(cid: str) -> VectorStore:
    if cid not in ss.stores:
        store = VectorStore()
        chunks, vecs = db.get_chunks(cid)
        if chunks:
            store.add(chunks, vecs)
        ss.stores[cid] = store
    return ss.stores[cid]


def index_pdf(cid: str, uploaded) -> None:
    chunks = chunk_pages(load_pdf(uploaded.getvalue()), uploaded.name)
    if chunks:
        db.add_chunks(cid, chunks, embed([c["text"] for c in chunks]))
    ss.stores.pop(cid, None)        # force reload from DB


def show_sources(hits: list[dict]) -> None:
    if not hits:
        return
    with st.expander(f"Sources ({len(hits)})"):
        for i, h in enumerate(hits, start=1):
            if h.get("kind") == "web":
                title = h["source"].replace("[", "(").replace("]", ")")
                st.markdown(f"**[{i}] [{title}]({h['url']})**  \n"
                            f"{urlparse(h['url']).netloc} · fetched {h.get('fetched', '')}")
            else:
                st.markdown(f"**[{i}] {h['source']}** · page {h['page']} · similarity {h['score']:.2f}")
            st.caption(h["text"][:400] + ("…" if len(h["text"]) > 400 else ""))


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### 💬 DocChat")
    st.button("＋  New chat", on_click=new_chat, use_container_width=True)

    st.caption("Recent chats")
    chats = db.list_chats()
    if not chats:
        st.caption("No chats yet.")
    for c in chats:
        left, right = st.columns([6, 1])
        title = c["title"] if len(c["title"]) <= 26 else c["title"][:25] + "…"
        left.button(title, key=f"chat_{c['id']}", on_click=select_chat, args=(c["id"],),
                    type="primary" if c["id"] == ss.chat_id else "secondary",
                    use_container_width=True)
        right.button("🗑", key=f"del_{c['id']}", on_click=delete_chat, args=(c["id"],))

    st.divider()
    docs = db.list_docs(ss.chat_id) if ss.chat_id else []
    if docs:
        st.markdown("**Documents in this chat**")
        for d in docs:
            st.caption(f"📄 {d['source']} · {d['chunks']} chunks")
    use_docs = st.toggle("Answer from documents", value=True, disabled=not docs,
                         help="Attach PDFs with the 📎 button in the message box.")
    use_web = st.toggle("🌐 Search the web", value=True,
                        help="Fetch live pages for each question and cite real links.")
    st.caption(f"Model: {MODEL}")

# ---------- main chat area ----------
messages = db.get_messages(ss.chat_id) if ss.chat_id else []
if not messages:
    st.markdown("## How can I help you today?")
    st.caption("Just start chatting. Use 📎 in the message box to attach a PDF, "
               "and the 🌐 switch in the sidebar for live web answers.")
for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        show_sources(m["sources"])

try:
    sub = st.chat_input("Message DocChat...", accept_file="multiple", file_type=["pdf"])
except TypeError:
    st.warning("Your Streamlit is too old for the attachment button. Run: pip install -U streamlit")
    sub = st.chat_input("Message DocChat...")

if sub:
    if isinstance(sub, str):
        text, files = sub.strip(), []
    else:
        text, files = (sub["text"] or "").strip(), list(sub["files"] or [])

    cid = ensure_chat()
    existing = {d["source"] for d in db.list_docs(cid)}
    new_files = [f for f in files if f.name not in existing]
    if new_files:
        with st.spinner("Indexing your PDF(s)..."):
            for f in new_files:
                index_pdf(cid, f)

    if db.get_chat(cid)["title"] == "New chat":
        db.rename_chat(cid, (text or files[0].name).replace("\n", " ")[:40])

    history = db.get_messages(cid)
    attach = "  \n".join(f"📎 {f.name}" for f in files)
    db.add_message(cid, "user", "\n\n".join(p for p in [text, attach] if p))

    if not text:    # attachment only: confirm and wait for a question
        names = ", ".join(f.name for f in files)
        db.add_message(cid, "assistant", f"Added **{names}**. What would you like to know about it?")
        st.rerun()

    with st.chat_message("user"):
        st.markdown("\n\n".join(p for p in [text, attach] if p))

    with st.chat_message("assistant"):
        doc_hits, web_hits = [], []
        store = get_store(cid)
        do_web = use_web and len(text.split()) >= 3        # skip search for "hi", "thanks"
        do_docs = use_docs and not store.is_empty()
        if do_docs:
            doc_hits = store.search(text, 3 if do_web else 4)
        if do_web:
            with st.spinner("🌐 Searching the web..."):
                web_hits, err = crawl(text, embed, k=3 if do_docs else 4)
            if err:
                st.caption(f"⚠️ Live web search unavailable ({err}). Answering without it.")
        hits = doc_hits + web_hits
        answer = st.write_stream(stream_answer(build_messages(text, hits, history)))
        show_sources(hits)

    db.add_message(cid, "assistant", answer, hits)
    st.rerun()      # refresh the sidebar so the new/updated chat shows up
