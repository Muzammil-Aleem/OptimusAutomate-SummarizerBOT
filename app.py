"""
app.py
------
Streamlit UI for the Document Summarizer & Q&A Bot.

Run with:  streamlit run app.py
"""

import os
import time
import streamlit as st

from core import document_loader, rag_engine, summarizer, qa

st.set_page_config(
    page_title="The Reading Room — Document Q&A",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------- THEME

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,500;0,600;1,400&family=Source+Sans+3:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root{
  --ink:#15211C;
  --ink-deep:#0E1714;
  --brass:#C9A227;
  --brass-dim:#8E7220;
  --parchment:#F4ECD8;
  --parchment-dim:#E9DEC2;
  --sage:#7FA08C;
  --text-light:#E9E4D6;
}

html, body, [class*="css"]  { font-family: 'Source Sans 3', sans-serif; }

.stApp{
  background: radial-gradient(ellipse at top left, #1C2920 0%, var(--ink) 45%, var(--ink-deep) 100%);
  color: var(--text-light) !important;
}

/* Force every native widget's text to the light color — Streamlit's own
   theme detection can otherwise render black text that's invisible
   against our dark background, regardless of the colors set above. */
.stApp, .stApp p, .stApp span, .stApp label, .stApp div,
.stMarkdown, .stCaption, .stRadio label, .stRadio div,
[data-testid="stWidgetLabel"] p, [data-testid="stTabs"] button p,
[data-testid="stMetricValue"], [data-testid="stCaptionContainer"]{
  color: var(--text-light) !important;
}

/* Sidebar */
section[data-testid="stSidebar"]{
  background: var(--ink-deep);
  border-right: 1px solid #2A372F;
}
section[data-testid="stSidebar"] *{ color: var(--text-light) !important; }

/* Headlines */
h1, h2, h3 { font-family:'Lora', serif !important; color: var(--parchment) !important; }
h1 { font-style: italic; letter-spacing: -.01em; }

.eyebrow{
  font-family:'IBM Plex Mono', monospace; font-size: 11px; letter-spacing:.18em;
  text-transform: uppercase; color: var(--brass); margin-bottom: 4px;
}

/* Parchment content card — explicitly overrides the global light-text
   rule above, since this card sits on a LIGHT background. */
.parch-card{
  background: var(--parchment);
  color: #2B2418 !important;
  border-radius: 2px;
  padding: 22px 26px;
  border-left: 4px solid var(--brass);
  box-shadow: 0 6px 18px rgba(0,0,0,.35);
  font-family: 'Source Sans 3', sans-serif;
  line-height: 1.65;
}
.parch-card, .parch-card p, .parch-card div, .parch-card span, .parch-card li{
  color: #2B2418 !important;
}
.parch-card h4{ font-family:'Lora', serif; margin-top:0; color:#2B2418 !important; }

/* Stat chips */
.chip-row{ display:flex; gap:10px; flex-wrap:wrap; margin: 6px 0 18px; }
.chip{
  font-family:'IBM Plex Mono', monospace; font-size:11.5px;
  border:1px solid var(--brass-dim); color: var(--brass) !important;
  padding: 5px 11px; border-radius: 20px; background: rgba(201,162,39,0.08);
}

/* Source citation tags */
.source-tag{
  display:inline-block; font-family:'IBM Plex Mono', monospace; font-size:10.5px;
  background: var(--sage); color: var(--ink-deep) !important; padding:3px 9px; border-radius:3px;
  margin: 3px 6px 3px 0;
}

hr{ border-color: #2A372F !important; }

/* Buttons */
.stButton button, .stDownloadButton button{
  background: var(--brass) !important; color: var(--ink-deep) !important;
  border: none !important; font-weight:600 !important;
  border-radius: 3px !important;
}
.stButton button:hover, .stDownloadButton button:hover{ background:#DDB73A !important; }

/* Chat bubbles */
div[data-testid="stChatMessage"]{
  background: rgba(244,236,216,0.06);
  border: 1px solid #2A372F;
  border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------- STATE

if "full_text" not in st.session_state:
    st.session_state.full_text = None
    st.session_state.chunks = None
    st.session_state.index = None
    st.session_state.summary = None
    st.session_state.summary_engine = None
    st.session_state.chat_history = []
    st.session_state.doc_name = None

# -------------------------------------------------------------------- SIDEBAR

with st.sidebar:
    st.markdown('<div class="eyebrow">The Reading Room</div>', unsafe_allow_html=True)
    st.title("📖 Document Desk")
    st.caption("Upload a document, get a summary, then interrogate it.")

    st.divider()
    uploaded = st.file_uploader("Upload a PDF or .txt file", type=["pdf", "txt"])

    st.markdown("**Retrieval settings**")
    chunk_size = st.slider("Chunk size (characters)", 400, 2000, 900, step=100)
    overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)
    top_k = st.slider("Chunks retrieved per question", 1, 10, 6)

    st.markdown("**Summary style**")
    style = st.radio("Format", ["brief", "detailed", "bullets"], horizontal=False, label_visibility="collapsed")

    st.divider()
    engine_note = "🟢 Claude API connected" if summarizer.ai_available() else "🟡 No API key — using offline fallback"
    st.caption(engine_note)
    backend_note = f"Retrieval backend: {rag_engine.SemanticIndex.name if rag_engine.semantic_available() else rag_engine.TfidfIndex.name}"
    st.caption(backend_note)

    if st.button("🗑️ Clear document & chat", use_container_width=True):
        for k in ["full_text", "chunks", "index", "summary", "summary_engine", "chat_history", "doc_name"]:
            st.session_state[k] = None if k != "chat_history" else []
        st.rerun()

# ---------------------------------------------------------------- PROCESS FILE

if uploaded is not None and uploaded.name != st.session_state.doc_name:
    with st.spinner("Reading and indexing document…"):
        file_bytes = uploaded.read()
        full_text, chunks = document_loader.load_and_chunk(
            file_bytes, uploaded.name, chunk_size=chunk_size, overlap=overlap
        )
        index = rag_engine.build_index(chunks)

        st.session_state.full_text = full_text
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.doc_name = uploaded.name
        st.session_state.summary = None
        st.session_state.chat_history = []

# -------------------------------------------------------------------- HEADER

st.markdown('<div class="eyebrow">Summarize · Retrieve · Answer</div>', unsafe_allow_html=True)
st.title("The Reading Room")
st.caption("A quiet desk for long documents — drop in a PDF or text file, get the gist, then ask it anything.")

if st.session_state.full_text is None:
    st.markdown("""
    <div class="parch-card">
      <h4>No document on the desk yet</h4>
      Upload a PDF or .txt file from the sidebar to begin. Once it's indexed you'll get
      an automatic summary and a chat box for follow-up questions, answered using
      retrieval-augmented generation grounded in the document's own text.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

word_count = len(st.session_state.full_text.split())
st.markdown(f"""
<div class="chip-row">
  <span class="chip">📄 {st.session_state.doc_name}</span>
  <span class="chip">{word_count:,} words</span>
  <span class="chip">{len(st.session_state.chunks)} chunks indexed</span>
</div>
""", unsafe_allow_html=True)

tab_summary, tab_qa = st.tabs(["📜 Summary", "💬 Ask Questions"])

# ------------------------------------------------------------------ SUMMARY TAB

with tab_summary:
    col1, col2 = st.columns([1, 5])
    with col1:
        generate = st.button("Generate summary", use_container_width=True)
    with col2:
        st.caption(f"Style: **{style}**  ·  Engine: {'Claude (map-reduce)' if summarizer.ai_available() else 'offline extractive fallback'}")

    if generate or (st.session_state.summary is None and st.session_state.full_text):
        progress = st.progress(0, text="Summarizing…")

        def _cb(done, total):
            progress.progress(done / total, text=f"Summarizing section {done}/{total}…")

        result = summarizer.summarize_document(
            st.session_state.full_text, st.session_state.chunks, style=style, progress_cb=_cb
        )
        progress.empty()
        st.session_state.summary = result["summary"]
        st.session_state.summary_engine = result["engine"]

    if st.session_state.summary:
        st.markdown(f"""
        <div class="parch-card">
          <h4>Summary</h4>
          {st.session_state.summary.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Generated via: {st.session_state.summary_engine}")
        st.download_button(
            "⬇ Download summary (.txt)",
            data=st.session_state.summary,
            file_name=f"{st.session_state.doc_name or 'document'}_summary.txt",
        )

# --------------------------------------------------------------------- QA TAB

with tab_qa:
    st.caption("Answers are grounded in the uploaded document via retrieval-augmented generation — "
               "the bot only sees the chunks most relevant to your question.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                tags = "".join(
                    f'<span class="source-tag">p.{s["page"] or "?"} · {s["score"]}</span>'
                    for s in msg["sources"]
                )
                st.markdown(tags, unsafe_allow_html=True)
                with st.expander("View the exact text the answer was grounded in"):
                    for i, s in enumerate(msg["sources"]):
                        st.markdown(f"**Source {i+1} — page {s['page'] or '?'} (relevance {s['score']})**")
                        st.text(s.get("full_text", s["preview"]))

    question = st.chat_input("Ask something about this document…")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the document…"):
                retrieved = st.session_state.index.query(question, k=top_k)
                result = qa.answer_question(question, retrieved, st.session_state.chat_history)
            st.markdown(result["answer"])
            if result["sources"]:
                tags = "".join(
                    f'<span class="source-tag">p.{s["page"] or "?"} · {s["score"]}</span>'
                    for s in result["sources"]
                )
                st.markdown(tags, unsafe_allow_html=True)
                with st.expander("View the exact text the answer was grounded in"):
                    for i, s in enumerate(result["sources"]):
                        st.markdown(f"**Source {i+1} — page {s['page'] or '?'} (relevance {s['score']})**")
                        st.text(s.get("full_text", s["preview"]))
            st.caption(f"Engine: {result['engine']}")

        st.session_state.chat_history.append({
            "role": "assistant", "content": result["answer"], "sources": result["sources"]
        })
