# How it works — Line by Line

---

## ingest.py

This script runs **once**. It reads your PDF, converts it to vectors, and saves everything to disk so `ask.py` can use it later.

```python
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import json
import faiss
```
Importing the libraries. `pypdf` reads PDFs, `sentence_transformers` converts text to vectors, `faiss` is the vector database, `numpy` handles array math, `json` saves the chunks.

---

```python
pdf_path = "doc.pdf"
reader = PdfReader(pdf_path)
```
Pointing to the PDF file and creating a reader object. The reader gives you access to each page.

---

```python
pages = [page.extract_text() for page in reader.pages]
```
Looping through every page in the PDF and pulling out the raw text. Result is a list of strings — one string per page.

---

```python
chunks = [p.strip() for p in pages if p and p.strip()]
```
Cleaning up. `.strip()` removes leading/trailing whitespace. The `if p and p.strip()` filters out empty pages (scanned images, blank pages) so we don't embed useless content.

---

```python
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)
```
Loading a small, fast embedding model (80MB). `.encode()` converts each chunk of text into a vector of 384 numbers. Similar text will produce similar vectors — this is what makes search work.

After this line, `embeddings` is a 2D array of shape `(num_chunks, 384)`.

---

```python
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
```
Reading the vector size (384) from the embeddings array. Creating a FAISS index:
- `Flat` — stores all vectors as-is, no compression
- `L2` — uses Euclidean distance to measure similarity (smaller = more similar)

Think of it as an empty box that knows how to find the closest vectors.

---

```python
index.add(np.array(embeddings, dtype="float32"))
```
Loading all the vectors into the FAISS index. FAISS requires `float32` so we cast it. After this, FAISS is ready to search.

---

```python
faiss.write_index(index, "index.faiss")
```
Saving the FAISS index to disk. Embedding is the slow step — we do it once and reuse the saved index every time we ask a question.

---

```python
with open("chunks.json", "w") as f:
    json.dump(chunks, f)
```
Saving the raw text chunks to disk. FAISS only stores vectors, not the original text. When FAISS says "chunk 4 is the best match", we need the actual text of chunk 4 to send to the LLM. That's what this file is for.

---

## ask.py

This script runs every time you want to ask a question. It loads the saved index, embeds your question, finds the closest chunks, and sends them to the LLM.

```python
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
```
Same libraries as before, plus `groq` for the LLM API.

---

```python
index = faiss.read_index("index.faiss")
with open("chunks.json", "r") as f:
    chunks = json.load(f)
```
Loading the FAISS index and text chunks that were saved by `ingest.py`. These two always go together — the index gives you positions, the chunks give you the actual text.

---

```python
model = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key="your_key")
```
Loading the same embedding model (must be the same model used in ingest, otherwise vectors won't match). Creating the Groq client to call the LLM.

---

```python
def ask(query, top_k=3):
```
Defining the function that does the full RAG pipeline. `top_k=3` means we retrieve the 3 most relevant chunks.

---

```python
    q_embedding = model.encode([query])
```
Converting the user's question into a vector using the same model. We pass a list `[query]` because `.encode()` expects a list. Result is shape `(1, 384)`.

---

```python
    _, indices = index.search(np.array(q_embedding, dtype="float32"), top_k)
```
Searching FAISS for the 3 vectors closest to the question vector. Returns two things:
- `_` — the distances (we don't need them, so we discard with `_`)
- `indices` — which chunk numbers matched, e.g. `[[2, 7, 4]]`

It's a 2D array because FAISS supports searching multiple queries at once. We only have one question so we take `indices[0]`.

---

```python
    context = " ".join([chunks[i] for i in indices[0]])
```
Using the indices to fetch the actual text. If indices are `[2, 7, 4]`, we grab chunks 2, 7, and 4 and join them into one string. This becomes the context we give to the LLM.

---

```python
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context." + context
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )
```
Sending the question + context to the LLM. The `messages` list defines the conversation:
- `system` — instructions to the LLM + the retrieved context. By injecting the chunks here, we're telling the model exactly what to answer from.
- `user` — the actual question

The model reads the context from the system prompt and uses it to answer the user's question instead of relying on its own training data.

---

```python
    return response.choices[0].message.content
```
Extracting the text answer from the API response object.

---

```python
print("Welcome to RAG : Enter your query")
while True:
    query = input("Enter your query: ").strip()
    print("\nAnswer : ", ask(query), "\n\n")
```
A simple terminal loop. Takes input, calls `ask()`, prints the answer. Runs forever until you press `Ctrl+C`.

---

## The full flow in one picture

```
ingest.py
──────────
doc.pdf → extract text → pages[] → clean → chunks[]
                                               ↓
                                    embed with MiniLM
                                               ↓
                                    vectors (384-dim each)
                                               ↓
                              FAISS index ←── add vectors
                                    ↓                ↓
                             index.faiss         chunks.json


ask.py
──────────
user types question
        ↓
embed question → question vector
        ↓
FAISS search → top 3 chunk indices
        ↓
fetch chunks[i] → context string
        ↓
Groq API (system: context, user: question)
        ↓
answer printed in terminal
```
