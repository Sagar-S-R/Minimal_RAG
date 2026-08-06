# Minimal RAG Pipeline

A minimal Retrieval-Augmented Generation (RAG) system that lets you chat with any PDF from your terminal. Built with FAISS, SentenceTransformers, and Groq.

---

## What it does

You drop a PDF in the folder, run one script to index it, then ask questions about it in the terminal. The system finds the most relevant parts of the PDF and sends them to an LLM to generate an answer.

---

## Project Structure

```
├── doc.pdf          ← your PDF goes here (must be named doc.pdf)
├── ingest.py        ← run once to index the PDF
├── ask.py           ← run to ask questions
├── requirements.txt
├── index.faiss      ← created after running ingest.py
└── chunks.json      ← created after running ingest.py
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Sagar-S-R/Minimal_RAG.git
cd Minimal_RAG
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Get a Groq API key**

Go to [console.groq.com](https://console.groq.com) → API Keys → Create key.

**4. Add your API key to `ask.py`**
```python
client = Groq(api_key="your_groq_api_key_here")
```

---

## Usage

**Step 1 — Add your PDF**

Place your PDF in the project root and rename it to `doc.pdf`.

**Step 2 — Index the PDF (run once)**
```bash
python ingest.py
```
This creates `index.faiss` and `chunks.json`. You only need to run this again if you change the PDF.

**Step 3 — Ask questions**
```bash
python ask.py
```
```
Welcome to RAG : Enter your query
Enter your query: what is this document about?
Answer: ...
```
Press `Ctrl+C` to exit.

---

## Stack

| Component | Library | Purpose |
|---|---|---|
| PDF parsing | pypdf | Extract text from PDF |
| Embedding | sentence-transformers | Convert text to vectors |
| Vector search | FAISS | Find relevant chunks |
| LLM | Groq (LLaMA) | Generate the final answer |

---

## How it works

```
PDF → text chunks → embeddings → FAISS index
                                      ↓
question → embed → search FAISS → top chunks → LLM → answer
```

See `EXPLANATION.md` for a line-by-line breakdown.
