import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings
import os
import csv
import tempfile
from pathlib import Path
import shutil
import time

# Import your constants
try:
    from constant import CHROMA_SETTINGS
except ImportError:
    class CHROMA_SETTINGS:
        persist_directory = "db"

# ===============================
# Custom Prompt Template
# ===============================
WASLA_PROMPT = PromptTemplate(
    template="""Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

# ===============================
# Dark Theme
# ===============================
def set_dark_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0B1020;
            color: #EAEAF2;
        }
        section.main > div {
            max-width: 900px;
            margin: auto;
        }
        h1 {
            text-align: center;
            font-size: 42px;
            font-weight: 700;
            color: #FFFFFF;
        }
        textarea {
            background-color: #111827;
            color: #E5E7EB;
            border-radius: 10px;
        }
        button {
            background-color: #2563EB !important;
            color: white !important;
            border-radius: 10px;
            width: 100%;
        }
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )

# ===============================
# Create Database from PDFs
# ===============================
def create_database_from_pdfs():
    """Create vector database from PDFs in docs folder"""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Check if docs folder exists
        status_text.text("📁 Checking for PDF files...")
        progress_bar.progress(10)
        
        docs_path = Path("docs")
        if not docs_path.exists():
            docs_path.mkdir(exist_ok=True)
            status_text.text("📁 Created docs folder. Please add PDFs to the 'docs' folder in your repository.")
            progress_bar.progress(100)
            time.sleep(2)
            return False
        
        # Find all PDFs
        pdf_files = list(docs_path.glob("**/*.pdf"))
        if not pdf_files:
            status_text.text("❌ No PDF files found in docs folder. Please add some PDFs.")
            progress_bar.progress(100)
            time.sleep(2)
            return False
        
        status_text.text(f"📚 Found {len(pdf_files)} PDF files")
        progress_bar.progress(20)
        
        # Step 2: Load PDFs
        all_documents = []
        for i, pdf_path in enumerate(pdf_files):
            status_text.text(f"📄 Loading: {pdf_path.name} ({i+1}/{len(pdf_files)})")
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            all_documents.extend(documents)
            progress_bar.progress(20 + int(30 * (i+1)/len(pdf_files)))
        
        status_text.text(f"✅ Loaded {len(all_documents)} pages")
        progress_bar.progress(50)
        
        # Step 3: Split documents
        status_text.text("✂️ Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        texts = text_splitter.split_documents(all_documents)
        status_text.text(f"✅ Created {len(texts)} text chunks")
        progress_bar.progress(70)
        
        # Step 4: Create embeddings and vector store
        status_text.text("🔤 Creating embeddings (this may take a few minutes)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"}
        )
        
        # Ensure db directory exists
        db_path = Path("db")
        db_path.mkdir(exist_ok=True)
        
        # Create vector store
        db = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=str(db_path)
        )
        
        status_text.text("💾 Saving vector store...")
        db.persist()
        progress_bar.progress(100)
        
        status_text.text("✅ Database created successfully!")
        time.sleep(2)
        return True
        
    except Exception as e:
        status_text.text(f"❌ Error: {str(e)}")
        progress_bar.progress(100)
        time.sleep(3)
        return False

# ===============================
# Initialize Vector Store
# ===============================
@st.cache_resource
def init_vectorstore():
    """Initialize the Chroma vector store"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"}
        )
        
        persist_dir = "db"
        
        # Check if vector store exists
        if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
            db = Chroma(
                embedding_function=embeddings,
                persist_directory=persist_dir
            )
            return db
        else:
            return None
            
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        return None

# ===============================
# Load LLM (Using Together AI)
# ===============================
@st.cache_resource
def load_llm():
    """Load LLM using API-based approach"""
    try:
        from langchain_together import Together
        
        llm = Together(
            model="meta-llama/Llama-2-7b-chat-hf",
            temperature=0.3,
            max_tokens=500,
            top_p=0.8,
            repetition_penalty=1.1,
            together_api_key=st.secrets["TOGETHER_API_KEY"]
        )
        return llm
        
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")
        return None

# ===============================
# Save to CSV
# ===============================
def save_to_csv(question, answer):
    """Save Q&A to CSV file"""
    csv_file = "chat_history.csv"
    file_exists = os.path.isfile(csv_file)
    
    try:
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Question", "Answer", "Timestamp"])
            import datetime
            writer.writerow([question, answer, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    except Exception as e:
        pass  # Silently fail for CSV

# ===============================
# Main App
# ===============================
def main():
    st.set_page_config(
        page_title="Wasla Solutions Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    set_dark_theme()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "db_initialized" not in st.session_state:
        st.session_state.db_initialized = False
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Wasla Solutions")
        st.markdown("---")
        
        # Database status and creation button
        st.subheader("📚 Database Status")
        
        db_path = Path("db")
        db_exists = (db_path / "chroma.sqlite3").exists()
        
        if db_exists:
            st.success("✅ Database ready")
            
            # Show database info
            try:
                size = sum(f.stat().st_size for f in db_path.glob('**/*') if f.is_file()) / (1024*1024)
                st.info(f"📊 Size: {size:.1f} MB")
            except:
                pass
        else:
            st.error("❌ Database not found")
            
            # Check for PDFs
            docs_path = Path("docs")
            pdf_files = list(docs_path.glob("**/*.pdf")) if docs_path.exists() else []
            
            if pdf_files:
                st.warning(f"📄 {len(pdf_files)} PDFs found")
                if st.button("🚀 Create Database Now", type="primary"):
                    with st.spinner("Creating database... This may take a few minutes."):
                        success = create_database_from_pdfs()
                        if success:
                            st.session_state.db_initialized = True
                            st.rerun()
            else:
                st.warning("📁 No PDFs found")
                st.info("Please add PDFs to the 'docs' folder in your GitHub repository")
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.title("💬 Wasla Solutions Chatbot")
    st.markdown("Ask questions about your PDF documents")
    
    # Initialize vectorstore if db exists
    if "vectorstore" not in st.session_state and db_exists:
        with st.spinner("🔄 Loading knowledge base..."):
            st.session_state.vectorstore = init_vectorstore()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Check if database exists
                    if not db_exists:
                        response = "⚠️ Please create the database first using the button in the sidebar."
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        return
                    
                    # Check if vectorstore is loaded
                    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
                        st.session_state.vectorstore = init_vectorstore()
                    
                    if st.session_state.vectorstore is None:
                        response = "⚠️ Could not load knowledge base. Please try recreating the database."
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        return
                    
                    # Load LLM
                    llm = load_llm()
                    
                    if llm:
                        # Create retriever
                        retriever = st.session_state.vectorstore.as_retriever(
                            search_kwargs={"k": 3}
                        )
                        
                        # Create QA chain
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            chain_type_kwargs={"prompt": WASLA_PROMPT},
                            return_source_documents=True
                        )
                        
                        # Get response
                        result = qa_chain.invoke({"query": prompt})
                        response = result['result']
                        
                        # Save to CSV
                        save_to_csv(prompt, response)
                        
                        # Show response
                        st.markdown(response)
                        
                        # Show sources in expander
                        with st.expander("📚 View Sources"):
                            for i, doc in enumerate(result['source_documents'], 1):
                                st.write(f"**Source {i}:**")
                                st.write(doc.page_content[:200] + "...")
                    else:
                        response = "⚠️ Could not load LLM. Please check API configuration."
                        st.markdown(response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()