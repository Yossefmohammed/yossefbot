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
from pathlib import Path
import time
import datetime
import re

# Import your constants
try:
    from constant import CHROMA_SETTINGS
except ImportError:
    class CHROMA_SETTINGS:
        persist_directory = "db"

# ===============================
# Enhanced Prompt Template
# ===============================
WASLA_PROMPT = PromptTemplate(
    template="""You are a knowledgeable assistant for Wasla Solutions. Your role is to provide helpful, accurate, and engaging responses based strictly on the provided context.

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the context provided above
2. If the context doesn't contain the answer, say: "I don't have information about that in my knowledge base. Could you ask something else about Wasla Solutions?"
3. Be conversational and professional, like a helpful colleague
4. Keep responses concise but informative
5. If relevant, mention specific details from the context
6. Use bullet points for lists when helpful
7. End by offering additional help
8. Vary your greetings - don't start every response the same way

YOUR RESPONSE:""",
    input_variables=["context", "question"]
)

# ===============================
# Dark Theme with Improvements
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
            margin-bottom: 0px;
        }
        .stChatMessage {
            background-color: #1E1E2E;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
            border: 1px solid #2D3748;
        }
        [data-testid="chatMessageContent"] {
            color: #E5E7EB !important;
        }
        textarea {
            background-color: #111827 !important;
            color: #E5E7EB !important;
            border-radius: 10px !important;
            border: 1px solid #2D3748 !important;
        }
        button {
            background-color: #2563EB !important;
            color: white !important;
            border-radius: 10px !important;
            width: 100%;
            transition: all 0.3s ease;
        }
        button:hover {
            background-color: #1D4ED8 !important;
            box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3);
        }
        .stExpander {
            background-color: #1E1E2E;
            border: 1px solid #2D3748;
            border-radius: 10px;
        }
        .stSpinner > div {
            border-color: #2563EB !important;
        }
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )

# ===============================
# Topic Extraction Helper
# ===============================
def extract_topics_from_docs(docs, max_topics=3):
    """Extract key topics from documents for better suggestions"""
    try:
        # Common Wasla-related keywords to look for
        topic_keywords = [
            "services", "solutions", "products", "digital", "platform",
            "fintech", "e-commerce", "marketing", "sales", "software",
            "development", "mobile", "web", "UI/UX", "branding",
            "consulting", "strategy", "innovation", "technology"
        ]
        
        # Combine text from first few docs
        all_text = " ".join([doc.page_content.lower() for doc in docs[:3]])
        
        # Find present keywords
        found_topics = [kw for kw in topic_keywords if kw in all_text]
        
        # Return unique topics (remove duplicates)
        return list(set(found_topics))[:max_topics]
    except:
        return ["services", "solutions", "digital transformation"]

# ===============================
# Load LLM using Groq (ENHANCED)
# ===============================
@st.cache_resource(ttl=3600)
def load_llm():
    """Load LLM using Groq's free API with enhanced prompting"""
    try:
        from groq import Groq
        
        # Check for API key in secrets
        if "GROQ_API_KEY" not in st.secrets:
            st.sidebar.error("❌ GROQ_API_KEY not found in secrets!")
            return None
        
        # Initialize Groq client
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        
        # Current working models (February 2026)
        working_models = [
            "llama-3.3-70b-versatile",           # Latest Llama (128K context)
            "deepseek-r1-distill-llama-70b",      # Reasoning model
            "meta-llama/llama-4-scout-17b-16e-instruct",  # Llama 4 (1M context)
            "gemma2-9b-it"                         # Google's model (fast)
        ]
        
        class GroqLLM:
            def __init__(self, client, model_name):
                self.client = client
                self.model = model_name
                self.model_info = {
                    "llama-3.3-70b-versatile": "Llama 3.3 70B",
                    "deepseek-r1-distill-llama-70b": "DeepSeek R1",
                    "meta-llama/llama-4-scout-17b-16e-instruct": "Llama 4 Scout",
                    "gemma2-9b-it": "Gemma 2 9B"
                }.get(model_name, model_name)
            
            def invoke(self, prompt):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": """You are Wasla AI, a knowledgeable and friendly assistant for Wasla Solutions. 

**Core Personality:**
- Professional, warm, and conversational
- Adapt your greeting style - don't repeat the same opening every time
- Be enthusiastic about helping users understand Wasla Solutions
- Precise and accurate, never inventing information

**Response Guidelines:**
1. **Vary your openings** - Use different greetings:
   - "I'd be happy to help with that!"
   - "Great question! Let me share what I know."
   - "Based on our documentation..."
   - (Only use "Great to have you here!" occasionally, not every time)

2. **When you have information:**
   - Present it clearly with bullet points when helpful
   - Reference specific details from context
   - End by offering additional help

3. **When information is missing:**
   - Acknowledge politely
   - Suggest related topics from your knowledge base
   - Keep it helpful, not repetitive

**Example when info exists:**
"Based on our documentation, Wasla Solutions specializes in [specific details]. This includes [list relevant services]. Would you like me to elaborate on any of these areas?"

**Example when info missing:**
"I don't have specific information about that in my knowledge base. However, I can tell you about [related topic 1], [related topic 2], or [related topic 3]. Which would be most helpful?"""},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.4,
                        max_tokens=600,
                        top_p=0.9
                    )
                    return completion.choices[0].message.content
                except Exception as e:
                    return f"I apologize, but I encountered an error: {str(e)}. Please try again."
        
        # Test each model until we find one that works
        status_placeholder = st.sidebar.empty()
        status_placeholder.info("🔄 Initializing Wasla AI...")
        
        for model_name in working_models:
            try:
                status_placeholder.info(f"🔄 Testing {model_name}...")
                test_llm = GroqLLM(client, model_name)
                test_response = test_llm.invoke("Say 'ready to help' briefly")
                
                if "Error" not in test_response and test_response and len(test_response) > 0:
                    status_placeholder.success(f"✅ Wasla AI ready with {test_llm.model_info}")
                    return test_llm
            except Exception as e:
                continue
        
        status_placeholder.error("❌ Could not initialize Wasla AI. Please check your API key.")
        return None
        
    except ImportError:
        st.sidebar.error("❌ Please install groq: pip install groq")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Setup error: {str(e)}")
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
            status_text.text("📁 Created docs folder. Please add PDFs to the 'docs' folder.")
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
@st.cache_resource(ttl=3600)
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
                writer.writerow(["Question", "Answer", "Time", "Date"])
            writer.writerow([question, answer, 
                           datetime.datetime.now().strftime("%H:%M:%S"), 
                           datetime.datetime.now().strftime("%Y-%m-%d")])
    except Exception as e:
        pass

# ===============================
# Process Question (ENHANCED with Topic Extraction)
# ===============================
def process_question(prompt, vectorstore, llm):
    """Process a single question with enhanced context handling and topic extraction"""
    try:
        # Create retriever with MMR for better diversity
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 15}
        )
        
        # Get relevant documents
        docs = retriever.get_relevant_documents(prompt)
        
        # Extract topics for better suggestions (if needed)
        topics = extract_topics_from_docs(docs)
        
        # Build context with source indication
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Document {i}]: {doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Add topic suggestions to prompt if this might be an unknown query
        enhanced_prompt = prompt
        if "?" in prompt and len(docs) < 2:  # If few relevant docs found
            topic_str = ", ".join(topics[:3])
            enhanced_prompt = f"""{prompt}

(Note: If you don't have specific information about this, please suggest these relevant topics that I can help with: {topic_str})"""
        
        # Format prompt with context
        formatted_prompt = WASLA_PROMPT.format(context=context, question=enhanced_prompt)
        
        # Get response from LLM
        response = llm.invoke(formatted_prompt)
        
        return response, docs
        
    except Exception as e:
        raise Exception(f"Error processing question: {str(e)}")

# ===============================
# Main App (ENHANCED)
# ===============================
def main():
    st.set_page_config(
        page_title="Wasla Solutions Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    set_dark_theme()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "llm" not in st.session_state:
        st.session_state.llm = None
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    
    # Add welcome message if not shown
    if "welcome_shown" not in st.session_state:
        st.session_state.welcome_shown = True
        st.session_state.messages.append({
            "role": "assistant", 
            "content": """👋 **Hello! I'm Wasla AI, your intelligent assistant for all things Wasla Solutions.**

I'm here to help you with:
- 📋 **Information** about Wasla Solutions services and offerings
- 🔍 **Details** from your documents and knowledge base
- 💡 **Answers** to your specific questions

**How I work:**
- I only provide information found in your documents
- I'll always show you my sources
- If I don't know something, I'll be honest about it

Feel free to ask me anything about the documents in my knowledge base. I'm ready to help! 🚀""",
            "sources": []
        })
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Wasla Solutions")
        st.markdown("---")
        
        # API Configuration
        st.subheader("🔑 API Configuration")
        
        # Show current API status
        if "GROQ_API_KEY" in st.secrets:
            st.success("✅ Groq API key configured")
            
            # Load LLM button
            if st.button("🔄 Initialize Wasla AI", use_container_width=True):
                with st.spinner("Loading AI model..."):
                    st.session_state.llm = load_llm()
                    if st.session_state.llm:
                        st.success("✅ Wasla AI initialized successfully!")
                        # Add system ready message
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "✅ **I'm ready to help!** You can now ask me questions about your documents. Feel free to ask about our services, solutions, or any specific information you need.",
                            "sources": []
                        })
                    else:
                        st.error("❌ Failed to initialize AI")
        else:
            st.error("❌ GROQ_API_KEY not found")
            st.info("""
            **To get a free Groq API key:**
            1. Go to [console.groq.com](https://console.groq.com)
            2. Sign up (free, no credit card)
            3. Copy your API key
            4. Add to secrets as GROQ_API_KEY
            """)
        
        st.markdown("---")
        
        # Database status and creation
        st.subheader("📚 Knowledge Base")
        
        db_path = Path("db")
        db_exists = (db_path / "chroma.sqlite3").exists()
        
        if db_exists:
            # Initialize vectorstore if not already done
            if st.session_state.vectorstore is None:
                with st.spinner("🔄 Loading knowledge base..."):
                    st.session_state.vectorstore = init_vectorstore()
            else:
                st.success("✅ Knowledge base ready")
                
            # Show database stats
            if st.session_state.vectorstore:
                try:
                    count = st.session_state.vectorstore._collection.count()
                    st.info(f"📊 {count} document chunks available")
                except:
                    pass
        else:
            st.warning("❌ Knowledge base not found")
            
            # Check for PDFs
            docs_path = Path("docs")
            pdf_files = list(docs_path.glob("**/*.pdf")) if docs_path.exists() else []
            
            if pdf_files:
                st.info(f"📄 {len(pdf_files)} PDFs found")
                if st.button("🚀 Create Knowledge Base", type="primary", use_container_width=True):
                    with st.spinner("Creating knowledge base... This may take a few minutes."):
                        success = create_database_from_pdfs()
                        if success:
                            st.success("✅ Knowledge base created!")
                            st.rerun()
            else:
                st.warning("📁 No PDFs found")
                st.info("Please add PDF files to the 'docs' folder")
        
        st.markdown("---")
        
        # Controls
        st.subheader("🎮 Controls")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset AI", use_container_width=True):
                st.cache_resource.clear()
                st.session_state.llm = None
                st.success("✅ AI reset")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                # Keep only the welcome message
                welcome_msg = st.session_state.messages[0] if st.session_state.messages and "welcome_shown" in st.session_state else None
                st.session_state.messages = [welcome_msg] if welcome_msg else []
                st.rerun()
        
        # System status
        with st.expander("ℹ️ System Status"):
            status_lines = []
            status_lines.append(f"🔑 Groq Key: {'✅' if 'GROQ_API_KEY' in st.secrets else '❌'}")
            status_lines.append(f"🤖 AI Model: {'✅' if st.session_state.llm else '❌'}")
            status_lines.append(f"📚 Knowledge: {'✅' if db_exists else '❌'}")
            status_lines.append(f"💬 Messages: {len(st.session_state.messages)}")
            st.write("\n".join(status_lines))
            
            # Export chat
            if os.path.exists("chat_history.csv"):
                with open("chat_history.csv", "r") as f:
                    st.download_button("📥 Export Chat", f, "chat_history.csv", use_container_width=True)
    
    # Main chat interface
    st.title("💬 Wasla AI Assistant")
    st.markdown("Ask me anything about your documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        preview = source[:200] + "..." if len(source) > 200 else source
                        st.write(f"**Source {i}:**")
                        st.write(preview)
    
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
                        response = "⚠️ **Knowledge base not found.** Please create one first using the button in the sidebar."
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response, "sources": []})
                        return
                    
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = init_vectorstore()
                    
                    if st.session_state.vectorstore is None:
                        response = "⚠️ **Could not load knowledge base.** Please try recreating it using the button in the sidebar."
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response, "sources": []})
                        return
                    
                    if st.session_state.llm is None:
                        with st.spinner("Initializing AI for the first time..."):
                            st.session_state.llm = load_llm()
                    
                    if st.session_state.llm is None:
                        response = "⚠️ **Could not initialize AI.** Please check your Groq API key in the sidebar and click 'Initialize Wasla AI'."
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response, "sources": []})
                        return
                    
                    # Process question with enhanced features
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
                        with st.expander("📚 View Sources"):
                            for i, doc in enumerate(sources, 1):
                                preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                                st.write(f"**Source {i}:**")
                                st.write(preview)
                    
                    # Add to session state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": [doc.page_content for doc in sources] if sources else []
                    })
                    
                except Exception as e:
                    error_msg = f"❌ **Error:** {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})

if __name__ == "__main__":
    main()