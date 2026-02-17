import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceEndpoint
from chromadb.config import Settings
import os
import csv
from pathlib import Path
import time
import requests

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
# Load LLM using Hugging Face (FIXED VERSION)
# ===============================
@st.cache_resource
def load_llm():
    """Load LLM using Hugging Face's free inference API"""
    
    # Use st.status for better progress tracking (Streamlit 1.27+)
    with st.status("🔄 Loading LLM...", expanded=False) as status:
        
        try:
            # Check if HF token exists
            if "HF_TOKEN" not in st.secrets:
                status.update(label="❌ HF_TOKEN not found!", state="error")
                st.error("Please add your Hugging Face token to Streamlit secrets")
                return None
            
            token = st.secrets["HF_TOKEN"]
            status.update(label=f"✅ Token found", state="running")
            
            # Test token validity with a simple API call
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(
                "https://huggingface.co/api/whoami",
                headers=headers,
                timeout=10
            )
            
            if response.status_code != 200:
                status.update(label="❌ Token invalid!", state="error")
                return None
            
            # Try direct InferenceClient (most reliable)
            try:
                from huggingface_hub import InferenceClient
                
                status.update(label="🔄 Connecting to Hugging Face...")
                
                client = InferenceClient(
                    model="meta-llama/Llama-2-7b-chat-hf",
                    token=token,
                    timeout=30
                )
                
                # Simple test
                status.update(label="🔄 Testing connection...")
                test_response = client.text_generation(
                    "Hello",
                    max_new_tokens=5,
                    temperature=0.3,
                )
                
                # Create a simple wrapper class
                class HuggingFaceLLM:
                    def __init__(self, client):
                        self.client = client
                    
                    def invoke(self, prompt):
                        try:
                            response = self.client.text_generation(
                                prompt,
                                max_new_tokens=500,
                                temperature=0.3,
                                top_p=0.8,
                                repetition_penalty=1.1,
                                do_sample=True,
                            )
                            return response
                        except Exception as e:
                            return f"Error: {str(e)}"
                    
                    def __call__(self, prompt):
                        return self.invoke(prompt)
                
                status.update(label="✅ LLM loaded successfully!", state="complete")
                return HuggingFaceLLM(client)
                
            except Exception as e:
                status.update(label=f"❌ Failed to load LLM: {str(e)}", state="error")
                return None
                
        except Exception as e:
            status.update(label=f"❌ Unexpected error: {str(e)}", state="error")
            return None

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
        
        db.persist()
        progress_bar.progress(100)
        
        status_text.text("✅ Database created successfully!")
        time.sleep(2)
        status_text.empty()
        progress_bar.empty()
        return True
        
    except Exception as e:
        status_text.text(f"❌ Error: {str(e)}")
        progress_bar.progress(100)
        time.sleep(3)
        status_text.empty()
        progress_bar.empty()
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
            
            # Test if it works
            try:
                count = db._collection.count()
                st.sidebar.success(f"✅ Database loaded with {count} documents")
            except:
                st.sidebar.warning("⚠️ Database loaded but may be empty")
            
            return db
        else:
            st.sidebar.warning("📁 No database found. Use the button below to create one.")
            return None
            
    except Exception as e:
        st.sidebar.error(f"❌ Error initializing vector store: {str(e)}")
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
# Process Question (separate function)
# ===============================
def process_question(prompt, vectorstore, llm):
    """Process a single question"""
    try:
        # Create retriever
        retriever = vectorstore.as_retriever(
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
        return result['result'], result['source_documents']
        
    except Exception as e:
        raise Exception(f"Error processing question: {str(e)}")

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
    
    if "llm" not in st.session_state:
        st.session_state.llm = None
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Wasla Solutions")
        st.markdown("---")
        
        # Hugging Face Token Status
        st.subheader("🔑 API Status")
        if "HF_TOKEN" in st.secrets:
            st.success("✅ Hugging Face token configured")
            
            # Add button to load/refresh LLM
            if st.button("🔄 Load LLM", use_container_width=True):
                with st.spinner("Loading LLM..."):
                    st.session_state.llm = load_llm()
                    if st.session_state.llm:
                        st.success("✅ LLM loaded successfully!")
                    else:
                        st.error("❌ Failed to load LLM")
        else:
            st.error("❌ HF_TOKEN not found in secrets")
            st.info("""
            To add your token:
            1. Get token from: https://huggingface.co/settings/tokens
            2. Add to Streamlit secrets as HF_TOKEN
            """)
        
        st.markdown("---")
        
        # Database status and creation
        st.subheader("📚 Database Status")
        
        db_path = Path("db")
        db_exists = (db_path / "chroma.sqlite3").exists()
        
        if db_exists:
            # Initialize vectorstore if not already done
            if st.session_state.vectorstore is None:
                with st.spinner("🔄 Loading database..."):
                    st.session_state.vectorstore = init_vectorstore()
            else:
                st.success("✅ Database ready")
        else:
            st.warning("❌ Database not found")
            
            # Check for PDFs
            docs_path = Path("docs")
            pdf_files = list(docs_path.glob("**/*.pdf")) if docs_path.exists() else []
            
            if pdf_files:
                st.info(f"📄 {len(pdf_files)} PDFs found")
                if st.button("🚀 Create Database Now", type="primary", use_container_width=True):
                    with st.spinner("Creating database... This may take a few minutes."):
                        success = create_database_from_pdfs()
                        if success:
                            st.rerun()
            else:
                st.warning("📁 No PDFs found")
                st.info("Please add PDF files to the 'docs' folder in your GitHub repository")
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Show status
        with st.expander("ℹ️ System Status"):
            status_lines = []
            status_lines.append(f"🔑 HF Token: {'✅' if 'HF_TOKEN' in st.secrets else '❌'}")
            status_lines.append(f"🤖 LLM: {'✅' if st.session_state.llm else '❌'}")
            status_lines.append(f"📚 Database: {'✅' if db_exists else '❌'}")
            status_lines.append(f"💬 Messages: {len(st.session_state.messages)}")
            st.write("\n".join(status_lines))
    
    # Main chat interface
    st.title("💬 Wasla Solutions Chatbot")
    st.markdown("Ask questions about your PDF documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**Source {i}:**")
                        st.write(source[:200] + "...")
    
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
                    # Check prerequisites
                    if not db_exists:
                        st.error("⚠️ Please create the database first using the button in the sidebar.")
                        return
                    
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = init_vectorstore()
                    
                    if st.session_state.vectorstore is None:
                        st.error("⚠️ Could not load knowledge base. Please try recreating the database.")
                        return
                    
                    if st.session_state.llm is None:
                        with st.spinner("Loading LLM for the first time..."):
                            st.session_state.llm = load_llm()
                    
                    if st.session_state.llm is None:
                        st.error("⚠️ Could not load LLM. Please check your Hugging Face token and click 'Load LLM' in the sidebar.")
                        return
                    
                    # Process question
                    response, sources = process_question(
                        prompt, 
                        st.session_state.vectorstore, 
                        st.session_state.llm
                    )
                    
                    # Save to CSV
                    save_to_csv(prompt, response)
                    
                    # Show response
                    st.markdown(response)
                    
                    # Show sources
                    if sources:
                        with st.expander("📚 Sources"):
                            for i, doc in enumerate(sources, 1):
                                st.write(f"**Source {i}:**")
                                st.write(doc.page_content[:200] + "...")
                    
                    # Add to session state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": [doc.page_content for doc in sources] if sources else []
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()