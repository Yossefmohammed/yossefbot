import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from chromadb.config import Settings
import os
import csv
import tempfile
from pathlib import Path
import shutil

# Import your constants
try:
    from constant import CHROMA_SETTINGS
except ImportError:
    # Fallback settings
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
        .stProgress > div > div > div > div {
            background-color: #2563EB;
        }
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )

# ===============================
# Initialize Vector Store
# ===============================
@st.cache_resource
def init_vectorstore():
    """Initialize the Chroma vector store"""
    try:
        # Use HuggingFaceEmbeddings instead of SentenceTransformerEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"}
        )
        
        # Ensure persist directory exists
        persist_dir = CHROMA_SETTINGS.persist_directory
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Check if vector store exists
        if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
            db = Chroma(
                embedding_function=embeddings,
                persist_directory=persist_dir
            )
            return db
        else:
            st.warning("⚠️ Vector store not found. Please run ingest.py locally first.")
            return None
            
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        return None

# ===============================
# Load LLM (Using Together AI for cloud)
# ===============================
@st.cache_resource
def load_llm():
    """Load LLM using API-based approach"""
    try:
        # Try Together AI first
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
        
    except (ImportError, KeyError):
        try:
            # Fallback to Replicate
            from langchain_community.llms import Replicate
            
            llm = Replicate(
                model="meta/llama-2-7b-chat:13c3cdee13ee059ab779f0291d29054dab00a47dad8261375654de5540165fb0",
                input={
                    "temperature": 0.3,
                    "max_length": 500,
                    "top_p": 0.8,
                    "repetition_penalty": 1.1
                },
                replicate_api_token=st.secrets["REPLICATE_API_TOKEN"]
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
        st.warning(f"Could not save to CSV: {str(e)}")

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
    
    if "vectorstore" not in st.session_state:
        with st.spinner("🔄 Loading knowledge base..."):
            st.session_state.vectorstore = init_vectorstore()
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=Wasla+Solutions", use_column_width=True)
        st.title("🤖 Chatbot Settings")
        
        st.markdown("---")
        
        # Show vector store status
        if st.session_state.vectorstore:
            st.success("✅ Knowledge base loaded")
            st.info(f"📚 Vector store location: {CHROMA_SETTINGS.persist_directory}")
        else:
            st.error("❌ Knowledge base not found")
            st.warning("Please run ingest.py locally first")
        
        st.markdown("---")
        
        # Add clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
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
                    if not st.session_state.vectorstore:
                        response = "⚠️ Knowledge base not loaded. Please run ingest.py locally first."
                    else:
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
                            
                            # Show sources
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