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
# Load LLM using Hugging Face (with debugging)
# ===============================
@st.cache_resource
def load_llm():
    """Load LLM using Hugging Face's free inference API with debugging"""
    
    # Add debug placeholder
    debug_info = st.empty()
    debug_info.info("🔍 Starting LLM loading process...")
    
    try:
        # Check if HF token exists
        debug_info.info("🔍 Checking for HF_TOKEN in secrets...")
        if "HF_TOKEN" not in st.secrets:
            debug_info.error("❌ HF_TOKEN not found in secrets!")
            st.error("Please add your Hugging Face token to Streamlit secrets")
            return None
        
        token = st.secrets["HF_TOKEN"]
        debug_info.info(f"✅ Token found (starts with: {token[:10]}...)")
        
        # Test token validity with a simple API call
        debug_info.info("🔍 Testing token validity...")
        import requests
        
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(
            "https://huggingface.co/api/whoami",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            user_info = response.json()
            debug_info.info(f"✅ Token valid! Logged in as: {user_info.get('name', 'Unknown')}")
        else:
            debug_info.error(f"❌ Token invalid! Status: {response.status_code}")
            return None
        
        # Try different approaches
        debug_info.info("🔍 Attempting to initialize HuggingFaceEndpoint...")
        
        # Approach 1: Try with endpoint_url
        try:
            from langchain_community.llms import HuggingFaceEndpoint
            
            debug_info.info("🔍 Creating HuggingFaceEndpoint...")
            
            llm = HuggingFaceEndpoint(
                endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf",
                huggingfacehub_api_token=token,
                task="text-generation",
                max_new_tokens=100,  # Smaller for testing
                temperature=0.3,
                top_p=0.8,
                do_sample=True,
            )
            
            # Test with a simple prompt
            debug_info.info("🔍 Testing with a simple prompt...")
            test_response = llm.invoke("Say 'Hello' in one word")
            debug_info.success(f"✅ LLM loaded! Test response: {test_response[:50]}...")
            
            # Clear debug info
            debug_info.empty()
            return llm
            
        except Exception as e1:
            debug_info.warning(f"⚠️ Approach 1 failed: {str(e1)}")
            
            # Approach 2: Try HuggingFaceHub
            try:
                from langchain_community.llms import HuggingFaceHub
                
                debug_info.info("🔍 Trying HuggingFaceHub approach...")
                
                llm = HuggingFaceHub(
                    repo_id="meta-llama/Llama-2-7b-chat-hf",
                    huggingfacehub_api_token=token,
                    model_kwargs={
                        "temperature": 0.3,
                        "max_new_tokens": 100,
                        "top_p": 0.8,
                    }
                )
                
                # Test with a simple prompt
                debug_info.info("🔍 Testing HuggingFaceHub...")
                test_response = llm.invoke("Say 'Hello' in one word")
                debug_info.success(f"✅ LLM loaded via Hub! Test response: {test_response[:50]}...")
                
                debug_info.empty()
                return llm
                
            except Exception as e2:
                debug_info.warning(f"⚠️ Approach 2 failed: {str(e2)}")
                
                # Approach 3: Direct InferenceClient
                try:
                    from huggingface_hub import InferenceClient
                    
                    debug_info.info("🔍 Trying direct InferenceClient...")
                    
                    client = InferenceClient(
                        model="meta-llama/Llama-2-7b-chat-hf",
                        token=token,
                    )
                    
                    # Simple test
                    debug_info.info("🔍 Testing direct client...")
                    test_response = client.text_generation(
                        "Say 'Hello' in one word",
                        max_new_tokens=10,
                        temperature=0.3,
                    )
                    
                    debug_info.success(f"✅ Direct client works! Test response: {test_response}")
                    
                    # Create a simple wrapper
                    class SimpleLLM:
                        def __init__(self, client):
                            self.client = client
                        
                        def invoke(self, prompt):
                            return self.client.text_generation(
                                prompt,
                                max_new_tokens=500,
                                temperature=0.3,
                                top_p=0.8,
                                repetition_penalty=1.1,
                                do_sample=True,
                            )
                    
                    debug_info.empty()
                    return SimpleLLM(client)
                    
                except Exception as e3:
                    debug_info.error(f"❌ All approaches failed!")
                    debug_info.error(f"Error 1: {str(e1)}")
                    debug_info.error(f"Error 2: {str(e2)}")
                    debug_info.error(f"Error 3: {str(e3)}")
                    return None
                    
    except Exception as e:
        debug_info.error(f"❌ Unexpected error: {str(e)}")
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
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Wasla Solutions")
        st.markdown("---")
        
        # Hugging Face Token Status
        st.subheader("🔑 API Status")
        if "HF_TOKEN" in st.secrets:
            st.success("✅ Hugging Face token configured")
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
            if "vectorstore" not in st.session_state:
                with st.spinner("🔄 Loading database..."):
                    st.session_state.vectorstore = init_vectorstore()
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
        
        # Show rate limits info
        with st.expander("ℹ️ About Free Tier"):
            st.write("""
            **Hugging Face Free Tier Limits:**
            - 30,000 input characters per month
            - Rate limited
            - Llama 2 7B model
            
            Perfect for demo and testing!
            """)
    
    # Main chat interface
    st.title("💬 Wasla Solutions Chatbot")
    st.markdown("Ask questions about your PDF documents")
    
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
                        response = "⚠️ Could not load LLM. Please check your Hugging Face token."
                        st.markdown(response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()