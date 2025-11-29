# Lura — Offline AI Engine

Lura is a fully offline Retrieval-Augmented Generation (RAG) system that allows you to ingest files, build a semantic vector index, and ask natural‑language questions — all without relying on any external API, cloud services, or internet connection.

It combines:
- **Sentence-Transformer embeddings** (MiniLM-L12)
- **FAISS vector search**
- **GGUF local LLM inference (Qwen 2.5 1.5B)** using `llama.cpp`
- A clean CLI workflow for ingestion, querying, and RAG‑based answering.


## Features

#### 1. Offline RAG Querying
Lura retrieves the most relevant chunks from your ingested documents and feeds them to a local LLM for contextual answers.

#### 2. Semantic Search
Search your entire ingested corpus using embeddings + FAISS.

#### 3. Deterministic Indexing
Lura stores metadata along with FAISS so that the index is fully reconstructible and consistent.

#### 4. Embedding Model Lock
If you attempt to ingest text using a different embedding model, Lura blocks it and asks you to rebuild the index — preventing silent corruption.

#### 5. Model-Agnostic LLM Loader
Drop any GGUF model as `models/model.gguf`, and Lura will load it through `llama.cpp`.


## Why Use Lura?

- No API fees
- No external dependencies
- Privacy-first: everything stays on your machine
- Ideal for learning embeddings, RAG, FAISS, and local LLM inference


## Requirements

#### Python
```
Python 3.10 or 3.11
```

#### Dependencies
Installed automatically via:
```
pip install -r requirements.txt
```

Includes:
- `sentence-transformers`
- `faiss-cpu`
- `llama-cpp-python`
- `numpy`
- `pdfplumber`
- `python-docx`
- `tqdm`


#### Model Requirements
---
#### Embedding Model

This project uses:

```
sentence-transformers/all-MiniLM-L12-v2
```

Download it once:

```
python -m sentence_transformers.download all-MiniLM-L12-v2
```

Place the transformer in `models/embeddings/all-MiniLM-L12-v2`

#### OR

#### If the download fails, run the code below:

```
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

model.save("models/embeddings/all-MiniLM-L12-v2")
```

After this, the model will live in your local HuggingFace cache and the system will run fully offline.

---

#### LLM Setup
#### Place your GGUF LLM here:

```
models/model.gguf
```

Recommended:
```
Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
```
Download [Qwen2.5-1.5B-Instruct-Q4_K_M.gguf](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/blob/main/qwen2.5-1.5b-instruct-q4_k_m.gguf)

## CLI Usage

Run the app:
```
python src/cli.py
```
#### 1. Ingest a File

Lura loads text → chunks it → embeds → stores into FAISS.

#### 2. Ingest Directory
Recursively ingests all `.txt`, `.md`, `.pdf`, `.docx`.

#### 3. Reset Vector Index
Wipes:
```
vector_index.faiss
vector_index.faiss.meta
```

Useful if:
- you changed embedding model
- index became corrupted
- you want a fresh start

#### 4. Semantic Query

This searches your ingested files by meaning, not exact words.

**Example**
```
Enter your query: benefits of exercise
```

Output:
```
[0] id:4 Score:0.91
    Text: Regular physical activity improves mood, strengthens the heart, and increases overall energy levels...
```
#### 5. RAG Question Answering

Retrieves relevant chunks and uses the local LLM to answer strictly from your ingested data.

**Example**
```
Enter your question: Based on the ingested notes, how do I change a flat tire?
```

Output:
```
Answer:
> Loosen the lug nuts, lift the car with a jack, remove the tire, mount the spare, tighten the nuts, and lower the car.

Press Enter to see sources
[1] id=5 score=0.84
"Step 1: Place the jack under the car frame near the flat tire..."
[2] id=6 score=0.77
"After securing the vehicle, remove the lug nuts and lift the tire off..."
```
#### 6. Exit
Shuts down the workflow.


## How Retrieval Works

##### 1. Lura embeds text using MiniLM-L12  
##### 2. FAISS stores normalized vectors  
##### 3. Query is embedded the same way  
##### 4. FAISS performs inner-product similarity  
##### 5. Lura applies filters
##### 6. Clean results are returned

These safeguards prevent hallucinations and force context-faithful answers.


## How RAG Works with the LLM

Lura formats a structured prompt:

```
Use ONLY the context below to answer the question.
If answer is not in context, say "I don't know."

### CONTEXT ###
[1] ...
[2] ...

### QUESTION ###
{your question}

### ANSWER ###
```

## Folder Structure
```
models/
│── embeddings/
     └── all-MiniLM-L12-v2/
            └──...
│── model.gguf
└── qwen2.5-1.5b-instruct-q4_k_m.gguf

src/
│── cli.py
│
├── encoder/
│     └── embedder.py
│
├── faiss/
│     └── ... (FAISS utilities)
│
├── inference/
│     └── local_llm.py
│
├── ingestion/
│     └── ... (file ingestion + chunking)
│
├── interface/
│     └── ... (CLI/UI hooks)
│
├── pipeline/
│     ├── pipeline.py
│     ├── rag.py
│     └── retrieve.py
│
└── storage/
      └── faiss_store.py

tests/
└── data.txt
``` 

## Notes
- Embedding model must remain consistent unless index is reset.
- LLM max context usage is limited by your GGUF + llama.cpp config.
- Large PDFs may produce many chunks — ingestion may take time.
- CPU-only inference is slower; optional GPU layers can be enabled.

### License
MIT License.