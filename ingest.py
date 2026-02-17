import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Constants for DB folder
class CHROMA_SETTINGS:
    persist_directory = "chroma"

def build_vectorstore():
    """Builds the Chroma vector store from PDFs in 'docs/'"""
    docs_folder = Path("docs")
    if not docs_folder.exists():
        print("❌ 'docs' folder not found. Please add your PDFs.")
        return None

    all_documents = []
    pdf_files = list(docs_folder.glob("**/*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in 'docs/' folder.")
        return None

    print(f"📚 Found {len(pdf_files)} PDF files")
    for pdf_path in pdf_files:
        print(f"📄 Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        all_documents.extend(documents)

    print(f"📊 Total pages loaded: {len(all_documents)}")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(all_documents)
    print(f"✂️ Created {len(texts)} text chunks")

    # Create embeddings
    print("🔤 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}
    )

    # Create vector store
    persist_dir = CHROMA_SETTINGS.persist_directory
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    print("💾 Creating vector store...")
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"✅ Vector store saved to: {persist_dir}")
    return db
