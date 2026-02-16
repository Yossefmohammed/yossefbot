import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from constant import CHROMA_SETTINGS
from langchain_community.embeddings import HuggingFaceEmbeddings  # Changed import
from pathlib import Path

def main():
    # Check if docs directory exists
    if not os.path.exists("docs"):
        print("❌ 'docs' folder not found. Please create it and add your PDFs.")
        return
    
    all_documents = []
    
    # Load PDFs
    pdf_files = list(Path("docs").glob("**/*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in docs folder.")
        return
    
    print(f"📚 Found {len(pdf_files)} PDF files")
    
    for pdf_path in pdf_files:
        print(f"📄 Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        all_documents.extend(documents)
    
    print(f"📊 Total pages loaded: {len(all_documents)}")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(all_documents)
    print(f"✂️ Created {len(texts)} text chunks")
    
    # Create embeddings
    print("🔤 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Create and persist vector store
    print("💾 Creating vector store...")
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=CHROMA_SETTINGS.persist_directory
    )
    
    print(f"✅ Ingestion completed! Vector store saved to: {CHROMA_SETTINGS.persist_directory}")

if __name__ == "__main__":
    main()