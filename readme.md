# RAG Document Assistant

A question-answering assistant that retrieves relevant 
information from documents using Retrieval-Augmented 
Generation (RAG), semantic search, and sentence transformers.

## What it does

The user uploads a document and asks questions about it.
The system chunks the document, converts chunks into vector 
embeddings, finds the most semantically similar chunks to 
the query, and generates a context-aware answer.

## How it works

1. **Document chunking** — splits text into overlapping chunks 
   to preserve context across boundaries
2. **Embedding** — encodes chunks using Sentence Transformers 
   (all-MiniLM-L6-v2)
3. **Retrieval** — finds top-k most similar chunks using 
   cosine similarity
4. **Generation** — produces an answer grounded in the 
   retrieved context

## Tech stack

- Python
- Sentence Transformers
- Vector embeddings (cosine similarity)
- Streamlit / CLI interface

## Run locally

git clone https://github.com/tamarataha/rag_document_assistant
cd rag_document_assistant
pip install -r requirements.txt
python app.py

## What I'd improve next

- PDF and multi-document support
- Replace cosine search with a proper vector DB (FAISS or ChromaDB)
- Connect to an LLM API for stronger generation
