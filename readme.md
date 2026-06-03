```markdown
# PDF RAG Pipeline – Islamic Studies Q&A (English-only Embeddings)

A retrieval‑augmented generation pipeline that answers questions **exclusively** from your uploaded PDF files.  
Uses a **state‑of‑the‑art English embedding model** (`all-mpnet-base-v2`) for vector search and **Google Gemini** for answer generation.  
Answers are neutral, scholarly, and cite sources directly from the PDFs.

## Repository

🔗 **GitHub:** [https://github.com/tariqnasheed/PDF_RAG_Milestone2.git](https://github.com/tariqnasheed/PDF_RAG_Milestone2.git)

## Important: Language Support
- The embedding model **only understands English**.  
- If your PDFs contain Arabic script (e.g., Quranic verses), those parts will **not** be indexed well.  
- For **bilingual (Arabic + English) retrieval**, use the multilingual version of this project (which uses `intfloat/multilingual-e5-large`).  
- This version is ideal when your PDFs are entirely in English, or you don't need Arabic‑language search.

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

### 2. Clone the Repository
```bash
git clone https://github.com/tariqnasheed/PDF_RAG_Milestone2.git
cd PDF_RAG_Milestone2
```

### 3. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Set Up Your API Key
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your-google-api-key-here
```
> ⚠️ **Important:** The `.env` file is listed in `.gitignore` and will **never** be pushed to GitHub. This keeps your API key secure.  
> An example file (`.env.example`) is provided in the repository to show you the required format without exposing real keys.

### 6. Add Your PDFs
Create a folder named `pdfs` in the project root and place your Quran/tafseer/history PDFs inside (subdirectories allowed).  
> The `pdfs/` folder is also listed in `.gitignore` to avoid uploading large or copyrighted files.

### 7. Run the Pipeline
```bash
python pdf_rag.py
```
The first run downloads the embedding model (~420 MB) and builds the vector store.  
Subsequent runs reuse the existing database.

## Files Excluded from Version Control (`.gitignore`)

The following files and folders are **intentionally excluded** from the GitHub repository to protect sensitive information and reduce repository size:

| Entry | Reason |
|-------|--------|
| `venv/` | Virtual environment – easily recreated with `requirements.txt` |
| `.env` | Contains your private `GOOGLE_API_KEY` |
| `pdfs/` | PDF files may be large or copyrighted |
| `chroma_db/` | Vector store is rebuilt locally from PDFs |
| `__pycache__/` | Python bytecode cache |
| `*.pyc` | Compiled Python files |
| `.DS_Store` | macOS system file |

> ✅ **What is committed:** Source code (`pdf_rag.py`), `requirements.txt`, `README.md`, and `.env.example` (template for API key).

## Embedding Model
- **Model:** `sentence-transformers/all-mpnet-base-v2`  
- **Dimensions:** 768  
- **Language:** English only  
- **Performance:** Top‑tier for semantic search in English.

If you later need multilingual support, replace the model name with `intfloat/multilingual-e5-large` and add the prefix wrapper (see the multilingual version of this project).

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `GOOGLE_API_KEY not found` | Check your `.env` file. Copy `.env.example` to `.env` and add your real key. |
| Empty pages / no text | Your PDFs are scanned images. Run OCR first (`ocrmypdf -l ara+eng input.pdf output.pdf`). |
| "The provided texts do not cover this topic" | The retrieved context does not contain the answer. Try increasing `k` (number of chunks) or adjusting `CHUNK_SIZE`/`CHUNK_OVERLAP`. |
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
- **Chat model** – switch `gemini-3.5-flash` to `gemini-3.5-pro` for more detailed answers, or to another provider (e.g., OpenAI) by replacing the chat model.

## Project Structure

```
PDF_RAG_Milestone2/
├── pdf_rag.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── pdfs/                  ← Not tracked by Git (add your PDFs here)
└── chroma_db/             ← Generated locally (not tracked)
```

## License

This project is for educational and personal use. Ensure you have the rights to any PDF content you use.

---