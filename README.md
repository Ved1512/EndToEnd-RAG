# 📚 End-to-End Modular RAG Pipeline

A production-grade, privacy-first, and completely free **Retrieval-Augmented Generation (RAG)** pipeline featuring a **FastAPI** backend orchestrator and a modern **React (Vite)** chat interface.

This system allows users to chat with multi-page PDFs locally and in real time, complete with document citations, vector similarity retrieval, and zero hallucinations.

---

## 🏛️ System Architecture

The pipeline is built on a clean **Separation of Concerns** principle. Every layer is modular, stateless where possible, and can be swapped without rewriting other components:

```
[ PDF Documents (/pdf) ]
          │
          ▼
[ 1. Ingestion Layer ] ────────► Recursive Token Chunking (tiktoken, 300 tokens, 50 overlap)
          │
          ▼
[ 2. Embedding Layer ] ────────► HuggingFace all-MiniLM-L6-v2 (384 dimensions, L2-normalized)
          │
          ▼
[ 3. Storage Layer ]   ────────► ChromaDB (HNSW Graph Vector Index + SQLite Metadata)
          │
          ▼
[ 4. API Orchestrator] ────────► FastAPI Endpoint (/chat) + CORS Middleware
          │
          ├────────────────────► Context Retrieval (Top-k Chunks)
          │
          ▼
[ 5. LLM Layer ]       ────────► Groq Cloud API (Llama-3.1-8b-instant, 128k context)
          │
          ▼
[ 6. Frontend UI ]     ────────► React (Vite) + Lucide Icons (Chatbot UI with Source Badges)
```

---

## 📖 Document Selection: Why *Crime and Punishment*? (A Trivial Decision)

For our primary test case, we placed Dostoevsky's classic novel **`crime-and-punishment.pdf`** into the `/pdf` directory.

### Why this choice was fundamentally trivial:
The entire pipeline is **strictly document-agnostic**. The ingestion engine does not care whether a PDF is:
* An enterprise financial quarterly report ($10-K$).
* A collection of complex medical research papers.
* An employee handbook or legal contract.
* Or a 500-page narrative novel like *Crime and Punishment*.

The pipeline treats any document as raw text streams, extracts page metadata, tokenizes the stream into discrete semantic packages, and stores them in geometric hyperspace. Any user can delete `crime-and-punishment.pdf`, drop in 20 company invoices, run `python ingest.py`, and the system adapts with zero code changes.

### Why testing with *Crime and Punishment* is illuminating:
While arbitrary, choosing a dense literary classic acts as an incredible benchmark to demonstrate the difference between **Local Retrieval** and **Global Synthesis**:
1. **Factual / Local Retrieval (RAG's Sweet Spot):** Asking *"What weapon did Raskolnikov use?"* or *"Who is the pawnbroker?"* retrieves the exact 1–2 pages from the novel in milliseconds, providing ground-truth answers with precise citations.
2. **Global Character Synthesis:** Asking *"Summarize Raskolnikov's character across the entire story"* highlights the classic RAG challenge: character development spans 500 pages, whereas top-$k$ vector retrieval only grabs the 3 most semantically similar paragraphs. This demonstrates why query phrasing and prompt engineering are vital in production RAG systems.

---

## 🛠️ Tech Stack & Technical Deep-Dive

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Orchestration** | LangChain Core / Community | Industry standard abstraction; allows hot-swapping models & vector databases. |
| **Chunking** | `RecursiveCharacterTextSplitter` | Token-based (via `tiktoken`) to strictly observe model context limits. |
| **Embeddings** | `all-MiniLM-L6-v2` (HuggingFace) | 100% free, runs locally on CPU/GPU, outputs dense 384-dimensional vectors. |
| **Vector DB** | ChromaDB (`hnswlib`) | Serverless, local persistence via SQLite & Parquet with HNSW graph indexing. |
| **LLM** | `llama-3.1-8b-instant` (Groq) | High reasoning quality, massive 128k context window, sub-second latency. |
| **Backend API**| FastAPI + Uvicorn | Async Python API framework with automatic validation and CORS support. |
| **Frontend UI**| React + Vite | Clean, responsive ChatGPT-style UI with automatic source document badges. |

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up the Python Backend

```bash
# Clone the repository
git clone https://github.com/Ved1512/EndToEnd-RAG.git
cd EndToEnd-RAG

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate   # Windows

# Install backend dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```
*(You can obtain a free API key at [console.groq.com](https://console.groq.com)).*

### 3. Ingest Your PDFs

Place any PDF documents you wish to analyze inside the `pdf/` folder, then run the ingestion manager:

```bash
python ingest.py
```
This script will:
1. Extract all text and page numbers from the PDFs.
2. Slice them into 300-token chunks with a 50-token overlap.
3. Pass them through `all-MiniLM-L6-v2` to compute 384-dimensional vectors.
4. Persist the vectors into a local `chroma_db/` folder.

### 4. Start the FastAPI Backend Server

```bash
uvicorn src.api.main:app --reload
```
* The backend will start on **`http://localhost:8000`**.
* Interactive API docs (Swagger UI) will be available at **`http://localhost:8000/docs`**.

### 5. Launch the React Frontend

Open a new terminal window:

```bash
cd frontend
npm install
npm run dev
```
* Open **`http://localhost:5173`** in your browser to start chatting with your documents!

---

## 🔒 Planned Enhancements

* [ ] **Role-Based Access Control (RBAC):** Implementing metadata filtering in ChromaDB (`allowed_roles`) coupled with JWT auth in FastAPI to restrict document access by user clearance.
* [ ] **Hierarchical Summarization:** Adding a Map-Reduce chain to handle global book-level summaries without context window degradation.
* [ ] **Streaming Responses:** Implementing Server-Sent Events (SSE) in FastAPI for token-by-token typing animations in the React UI.

---

## 📄 License

This project is licensed under the MIT License - open-source and free for commercial and non-commercial use.
