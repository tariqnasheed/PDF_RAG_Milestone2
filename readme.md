```markdown
# PDF RAG Pipeline – Islamic Studies Q&A (English-only Embeddings)

A retrieval‑augmented generation pipeline that answers questions **exclusively** from your uploaded PDF files.  
Uses a **state‑of‑the‑art English embedding model** (`all-mpnet-base-v2`) for vector search and **Google Gemini** for answer generation.  
Answers are neutral, scholarly, and cite sources directly from the PDFs.

## Important: Language Support
- The embedding model **only understands English**.  
- If your PDFs contain Arabic script (e.g., Quranic verses), those parts will **not** be indexed well.  
- For **bilingual (Arabic + English) retrieval**, use the multilingual version of this project (which uses `intfloat/multilingual-e5-large`).  
- This version is ideal when your PDFs are entirely in English, or you don’t need Arabic‑language search.

## Features

- **100% PDF‑grounded** – answers never rely on outside knowledge.
- **Source citations** – surah/verse, tafseer, book title/volume/page.
- **Persistent vector store** – Chroma database saved to `chroma_db/`.
- **All‑local embeddings** – the embedding model runs offline (no API calls for embeddings).
- **Google Gemini chat** – uses your `GOOGLE_API_KEY` (free tier available).

## Setup

### 1. Prerequisites
- Python 3.8+
- A Google API key ([get one here](https://makersuite.google.com/app/apikey))
- PDF files inside the `pdfs/` folder (must be **searchable**, not scanned images)

### 2. Installation
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 3. API Key
Create a `.env` file:
```
GOOGLE_API_KEY=your-key-here
```

### 4. Add your PDFs
Place your Quran/tafseer/history PDFs in the `pdfs/` folder (subdirectories allowed).

### 5. Run
```bash
python pdf_rag.py
```
The first run downloads the embedding model (~420 MB) and builds the vector store.  
Subsequent runs reuse the existing database.

## Embedding Model
- **Model:** `sentence-transformers/all-mpnet-base-v2`  
- **Dimensions:** 768  
- **Language:** English only  
- **Performance:** Top‑tier for semantic search in English.

If you later need multilingual support, replace the model name with `intfloat/multilingual-e5-large` and add the prefix wrapper (see the multilingual version of this project).

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `GOOGLE_API_KEY not found` | Check your `.env` file. |
| Empty pages / no text | Your PDFs are scanned images. Run OCR first (`ocrmypdf -l ara+eng input.pdf output.pdf`). |
| “The provided texts do not cover this topic” | The retrieved context does not contain the answer. Try increasing `k` (number of chunks) or adjusting `CHUNK_SIZE`/`CHUNK_OVERLAP`. |
| Retrieval misses English terms inside Arabic documents | The English‑only model may not match Arabic‑scripted content. Switch to a multilingual model for better cross‑lingual retrieval. |
| Quota errors (Gemini) | Wait or upgrade to a paid plan. The free tier has daily limits. |

## Configuration

Edit these variables at the top of `pdf_rag.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | 1500 | Characters per chunk |
| `CHUNK_OVERLAP` | 400 | Overlap between chunks |
| `k` | 5 | Number of chunks retrieved per question |

## Customisation

- **Embedding model** – change `model_name` in `create_vectorstore()` to any HuggingFace sentence‑transformer model.
- **Answer style** – modify the `template` in `create_rag_chain()`.
- **Chat model** – switch `gemini-2.5-flash` to `gemini-2.5-pro` for more detailed answers, or to another provider (e.g., OpenAI) by replacing the chat model.

```