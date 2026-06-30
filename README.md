# The Reading Room — Document Summarizer & Q&A Bot

Upload a PDF or text file, get an automatic summary, then ask it
questions in a chat box — answers are grounded in the document itself
using retrieval-augmented generation (RAG), not the model's general
knowledge.

```
doc-qa-bot/
├── app.py                     Streamlit UI (upload, summary tab, chat Q&A tab)
├── core/
│   ├── document_loader.py     PDF/txt extraction + overlapping chunking
│   ├── rag_engine.py          Retrieval: TF-IDF (default) or real embeddings if installed
│   ├── summarizer.py          Map-reduce summarization via Claude (+ offline fallback)
│   └── qa.py                  RAG-grounded question answering (+ offline fallback)
└── requirements.txt
```

## What it does

1. **Upload** a PDF or `.txt` file from the sidebar. It's parsed and
   split into overlapping chunks (page numbers are tracked for PDFs).
2. **Retrieve** — chunks are indexed for search. By default this uses a
   lightweight TF-IDF + cosine-similarity index (instant, no model
   download, works fully offline). If you install
   `sentence-transformers`, the app automatically switches to real
   semantic embeddings instead — no code changes needed.
3. **Summarize** — the document is summarized using a map-reduce
   approach: each section gets summarized individually via Claude, then
   those partial summaries are combined into one coherent final summary
   in your chosen style (brief / detailed / bullet list).
4. **Ask questions** — in the chat tab, each question retrieves the
   most relevant chunks from the index and Claude is asked to answer
   *using only those excerpts*, citing source pages. If it's not in the
   document, it says so instead of guessing.
5. **No API key? No problem** — both summarization and Q&A have
   dependency-light offline fallbacks (word-frequency extractive
   summarization, and direct excerpt retrieval) so the whole app works
   immediately, end to end, with zero configuration.

## Setup (run on your own PC)

**Requirements:** Python 3.10+

```bash
cd doc-qa-bot
pip install -r requirements.txt
```

> **Windows / PowerShell note:** if `pip install` fails compiling a
> package from source (a Rust/MSVC linker error), it's usually a
> Python-version-vs-prebuilt-wheel mismatch. Try updating Python to
> 3.12/3.13, or install "Build Tools for Visual Studio" with the
> "Desktop development with C++" workload.

### (Optional, recommended) Enable real AI summarization & Q&A

Without an API key the app still works end-to-end using offline
fallbacks. To use actual Claude-powered summaries and grounded answers,
get a key from https://console.anthropic.com and set it as an
environment variable **before** starting the app:

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-...your-key..."

# Windows PowerShell (current session only)
$env:ANTHROPIC_API_KEY="sk-ant-...your-key..."

# Windows PowerShell (persists across sessions — reopen PowerShell after running this)
setx ANTHROPIC_API_KEY "sk-ant-...your-key..."
```

### (Optional) Upgrade retrieval to real semantic embeddings

By default the app uses TF-IDF retrieval, which is fast and needs no
downloads but is keyword-based rather than truly semantic. For better
accuracy on documents where questions are phrased very differently from
the source text, install:

```bash
pip install sentence-transformers
```

The app detects this automatically on next launch and switches the
retrieval backend — check the sidebar, it tells you which backend is
active.

### Start the app

```bash
streamlit run app.py
```

This opens automatically in your browser at **http://localhost:8501**.
If it doesn't, open that URL manually.

## Using it

1. Upload a PDF or `.txt` file from the sidebar.
2. Adjust chunk size/overlap and how many chunks get retrieved per
   question if you want (defaults work fine for most documents).
3. Go to the **Summary** tab and click *Generate summary* (or it
   generates automatically on first upload). Pick a style: brief,
   detailed, or bullet list. Download it as `.txt` if you want.
4. Go to the **Ask Questions** tab and start chatting. Each answer
   shows which page(s) it pulled from and a relevance score, so you can
   verify the source yourself.
5. Upload a new file any time to start over, or use the *Clear
   document & chat* button in the sidebar.

## Notes

- Everything runs in-memory for the current Streamlit session —
  closing the tab or restarting the app clears the document and chat.
- PDF extraction quality depends on the PDF itself: scanned/image-only
  PDFs (no embedded text layer) won't extract anything useful here,
  since this performs text extraction, not OCR.
- For very large documents (100+ pages), map-reduce summarization makes
  multiple Claude API calls in sequence — this is normal and the
  progress bar shows section-by-section progress.
