import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import json
import faiss

pdf_path = "doc.pdf"
reader = PdfReader(pdf_path)

pages = [pages.extract_text() for pages in reader.pages]
chunks = [p.strip() for p in pages if p and p.strip()]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype="float32"))

faiss.write_index(index, "index.faiss")
with open("chunks.json", "w") as f:
    json.dump(chunks, f)
