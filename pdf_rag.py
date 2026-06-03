# pdf_rag.py
# PDF RAG Pipeline – Prophet S.W.S History Q&A (English-only embeddings)
# Loads PDFs from the "pdfs/" folder, splits into chunks,
# creates a Chroma vector store using a HuggingFace English embedding model,
# and answers questions using Gemini (grounded ONLY in the PDFs).

import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -------------------------------------------------------------------
# 1. Load environment variables
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY not found. Set it in .env file.")
    sys.exit(1)

# -------------------------------------------------------------------
# 2. Configuration
PDF_DIRECTORY = "pdfs"
CHROMA_PERSIST_DIR = "chroma_db"
CHUNK_SIZE = 1500                  # Characters per chunk
CHUNK_OVERLAP = 400                # Overlap between chunks
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 3. Load PDFs from the given directory
def load_pdfs(directory: str) -> list:
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Directory '{directory}' does not exist. Create it and add PDF files."
        )
    loader = DirectoryLoader(
        directory,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from '{directory}'.")
    return documents

# -------------------------------------------------------------------
# 4. Split documents into smaller chunks
def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} pages into {len(chunks)} chunks.")
    return chunks

# -------------------------------------------------------------------
# 5. Create Chroma vector store using an English-only embedding model
def create_vectorstore(chunks: list, persist_directory: str) -> Chroma:
    """
    Uses 'all-mpnet-base-v2', one of the strongest English sentence embedding models.
    It does **not** support Arabic – for bilingual texts, use a multilingual model.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"Vector store saved to '{persist_directory}' "
          f"with {vectordb._collection.count()} entries.")
    return vectordb

# -------------------------------------------------------------------
# 6. Build the RAG chain
def create_rag_chain(vectordb: Chroma):
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}          # Retrieve 5 most relevant chunks
    )

    template = """You are a knowledgeable Islamic studies assistant.
Answer the following question based SOLELY on the provided context from the uploaded PDFs
(Quran translation/tafseer and Islamic history books).
If the context does not contain enough information, say: "The provided texts do not cover this topic."
When answering, cite the specific source as it appears in the PDFs:
- For Quranic verses: mention the surah name and verse number if available.
- For tafseer: mention the tafseer name and page/section if present.
- For history books: mention the book title, volume, chapter, or page number if given.
Be concise and factual.

Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# -------------------------------------------------------------------
# 7. Main
def main():
    print("=== PDF RAG Pipeline – Life of Prophet S.W.S Q&A (English Embeddings) ===")

    documents = load_pdfs(PDF_DIRECTORY)
    if not documents:
        print("No documents found. Exiting.")
        return

    chunks = split_documents(documents)

    vectordb = create_vectorstore(chunks, CHROMA_PERSIST_DIR)

    rag_chain = create_rag_chain(vectordb)

    print("\nYou can now ask questions based on the PDFs. Type 'exit' to quit.")
    while True:
        question = input("\nYour question: ")
        if question.lower() == "exit":
            break
        if not question.strip():
            continue
        try:
            answer = rag_chain.invoke(question)
            print(f"\nAnswer: {answer}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()