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
    try:
        topic_keywords = [
            "services", "solutions", "products", "digital", "platform",
            "fintech", "e-commerce", "marketing", "sales", "software",
            "development", "mobile", "web", "UI/UX", "branding",
            "consulting", "strategy", "innovation", "technology"
        ]
        all_text = " ".join([doc.page_content.lower() for doc in docs[:3]])
        found_topics = [kw for kw in topic_keywords if kw in all_text]
        return list(set(found_topics))[:max_topics]
    except:
        return ["services", "solutions", "digital transformation"]

# ===============================
# Load LLM using Groq
# ===============================
@st.cache_resource(ttl=3600)
def load_llm():
    try:
        from groq import Groq
        if "GROQ_API_KEY" not in st.secrets:
            st.sidebar.error("❌ GROQ_API_KEY not found!"); return None
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        working_models = [
            "llama-3.3-70b-versatile",
            "deepseek-r1-distill-llama-70b",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "gemma2-9b-it"
        ]
        class GroqLLM:
            def __init__(self, client, model_name):
                self.client = client
                self.model = model_name
            def invoke(self, prompt):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": """You are Wasla AI, a knowledgeable assistant. Respond professionally, be precise, vary greetings, show sources if available."""},
                                  {"role": "user", "content": prompt}],
                        temperature=0.4, max_tokens=600, top_p=0.9
                    )
                    return completion.choices[0].message.content
                except Exception as e:
                    return f"I encountered an error: {str(e)}"
        status_placeholder = st.sidebar.empty()
        status_placeholder.info("🔄 Initializing Wasla AI...")
        for model_name in working_models:
            try:
                status_placeholder.info(f"🔄 Testing {model_name}...")
                test_llm = GroqLLM(client, model_name)
                test_response = test_llm.invoke("Say 'ready to help'")
                if "Error" not in test_response and test_response and len(test_response) > 0:
                    status_placeholder.success(f"✅ Wasla AI ready with {model_name}")
                    return test_llm
            except:
                continue
        status_placeholder.error("❌ Could not initialize Wasla AI. Check your API key.")
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
    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        status_text.text("📁 Checking for PDF files...")
        progress_bar.progress(10)
        docs_path = Path("docs")
        if not docs_path.exists(): docs_path.mkdir(exist_ok=True); status_text.text("📁 Created 'docs' folder. Please add PDFs."); progress_bar.progress(100); time.sleep(2); return False
        pdf_files = list(docs_path.glob("**/*.pdf"))
        if not pdf_files: status_text.text("❌ No PDF files found."); progress_bar.progress(100); time.sleep(2); return False
        status_text.text(f"📚 Found {len(pdf_files)} PDF files"); progress_bar.progress(20)
        all_documents = []
        for i, pdf_path in enumerate(pdf_files):
            status_text.text(f"📄 Loading: {pdf_path.name} ({i+1}/{len(pdf_files)})")
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            all_documents.extend(documents)
            progress_bar.progress(20 + int(30*(i+1)/len(pdf_files)))
        status_text.text(f"✅ Loaded {len(all_documents)} pages"); progress_bar.progress(50)
        status_text.text("✂️ Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(all_documents)
        status_text.text(f"✅ Created {len(texts)} text chunks"); progress_bar.progress(70)
        status_text.text("🔤 Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})
        db_path = Path("db"); db_path.mkdir(exist_ok=True)
        db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=str(db_path))
        db.persist(); progress_bar.progress(100)
        status_text.text("✅ Database created successfully!"); time.sleep(2); status_text.empty(); progress_bar.empty()
        return True
    except Exception as e:
        status_text.text(f"❌ Error: {str(e)}"); progress_bar.progress(100); time.sleep(3); status_text.empty(); progress_bar.empty(); return False

# ===============================
# Initialize Vector Store
# ===============================
@st.cache_resource(ttl=3600)
def init_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})
        persist_dir = "db"
        if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
            db = Chroma(embedding_function=embeddings, persist_directory=persist_dir)
            try: count = db._collection.count(); st.sidebar.success(f"✅ Database loaded with {count} documents")
            except: st.sidebar.warning("⚠️ Database loaded but may be empty")
            return db
        else: st.sidebar.warning("📁 No database found. Use the button below to create one."); return None
    except Exception as e:
        st.sidebar.error(f"❌ Error initializing vector store: {str(e)}"); return None

# ===============================
# Save to CSV
# ===============================
def save_to_csv(question, answer):
    csv_file = "chat_history.csv"
    file_exists = os.path.isfile(csv_file)
    try:
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(["Question","Answer","Time","Date"])
            writer.writerow([question, answer, datetime.datetime.now().strftime("%H:%M:%S"), datetime.datetime.now().strftime("%Y-%m-%d")])
    except: pass

# ===============================
# Process Question
# ===============================
def process_question(prompt, vectorstore, llm):
    try:
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":5,"fetch_k":15})
        docs = retriever.get_relevant_documents(prompt)
        topics = extract_topics_from_docs(docs)
        context_parts = [f"[Document {i}]: {doc.page_content}" for i, doc in enumerate(docs,1)]
        context = "\n\n".join(context_parts)
        enhanced_prompt = prompt
        if "?" in prompt and len(docs)<2:
            topic_str = ", ".join(topics[:3])
            enhanced_prompt = f"""{prompt}\n\n(Note: If you don't have specific information about this, please suggest these relevant topics: {topic_str})"""
        formatted_prompt = WASLA_PROMPT.format(context=context, question=enhanced_prompt)
        response = llm.invoke(formatted_prompt)
        return response, docs
    except Exception as e:
        raise Exception(f"Error processing question: {str(e)}")

# ===============================
# Main App
# ===============================
def main():
    st.set_page_config(page_title="Wasla Solutions Chatbot", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
    set_dark_theme()

    if "messages" not in st.session_state: st.session_state.messages=[]
    if "llm" not in st.session_state: st.session_state.llm=None
    if "vectorstore" not in st.session_state: st.session_state.vectorstore=None

    if "welcome_shown" not in st.session_state:
        st.session_state.welcome_shown=True
        st.session_state.messages.append({"role":"assistant","content":"""👋 **Hello! I'm Wasla AI, your intelligent assistant for all things Wasla Solutions.**\n\nI'm here to help you with:\n- 📋 **Information** about Wasla Solutions services and offerings\n- 🔍 **Details** from your documents and knowledge base\n- 💡 **Answers** to your specific questions\n\n**How I work:**\n- I only provide information found in your documents\n- I'll always show you my sources\n- If I don't know something, I'll be honest about it\n\nFeel free to ask me anything about the documents in my knowledge base. I'm ready to help! 🚀""","sources":[]})

    # -----------------------------
    # Sidebar
    # -----------------------------
    with st.sidebar:
        st.title("🤖 Wasla Solutions")
        st.markdown("---")
        st.subheader("🔑 API Configuration")
        if "GROQ_API_KEY" in st.secrets:
            st.success("✅ Groq API key configured")
            if st.button("🔄 Initialize Wasla AI", use_container_width=True):
                with st.spinner("Loading AI model..."):
                    st.session_state.llm = load_llm()
                    if st.session_state.llm:
                        st.success("✅ Wasla AI initialized successfully!")
                        st.session_state.messages.append({"role":"assistant","content":"✅ **I'm ready to help!** You can now ask me questions about your documents. Feel free to ask about our services, solutions, or any specific information you need.","sources":[]})
                    else: st.error("❌ Failed to initialize AI")
        else:
            st.error("❌ GROQ_API_KEY not found")
            st.info("""**To get a free Groq API key:**\n1. Go to [console.groq.com](https://console.groq.com)\n2. Sign up (free, no credit card)\n3. Copy your API key\n4. Add to secrets as GROQ_API_KEY""")

        st.markdown("---")
        st.subheader("📚 Knowledge Base")
        db_path = Path("db"); db_exists = (db_path / "chroma.sqlite3").exists()
        if db_exists:
            if st.session_state.vectorstore is None:
                with st.spinner("🔄 Loading knowledge base..."):
                    st.session_state.vectorstore = init_vectorstore()
            else: st.success("✅ Knowledge base ready")
            if st.session_state.vectorstore:
                try: count = st.session_state.vectorstore._collection.count(); st.info(f"📊 {count} document chunks available")
                except: pass
        else:
            st.warning("❌ Knowledge base not found")
            docs_path = Path("docs"); pdf_files = list(docs_path.glob("**/*.pdf")) if docs_path.exists() else []
            if pdf_files:
                st.info(f"📄 {len(pdf_files)} PDFs found")
                if st.button("🚀 Create Knowledge Base", type="primary", use_container_width=True):
                    with st.spinner("Creating knowledge base... This may take a few minutes."):
                        success=create_database_from_pdfs()
                        if success: st.success("✅ Knowledge base created!"); st.rerun()
            else:
                st.warning("📁 No PDFs found"); st.info("Please add PDF files to the 'docs' folder")

        st.markdown("---")
        st.subheader("🎮 Controls")
        col1,col2=st.columns(2)
        with col1:
            if st.button("🔄 Reset AI", use_container_width=True):
                st.cache_resource.clear(); st.session_state.llm=None; st.success("✅ AI reset"); st.rerun()
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                welcome_msg = st.session_state.messages[0] if st.session_state.messages and "welcome_shown" in st.session_state else None
                st.session_state.messages=[welcome_msg] if welcome_msg else []; st.rerun()
        with st.expander("ℹ️ System Status"):
            status_lines=[]
            status_lines.append(f"🔑 Groq Key: {'✅' if 'GROQ_API_KEY' in st.secrets else '❌'}")
            status_lines.append(f"🤖 AI Model: {'✅' if st.session_state.llm else '❌'}")
            status_lines.append(f"📚 Knowledge: {'✅' if db_exists else '❌'}")
            status_lines.append(f"💬 Messages: {len(st.session_state.messages)}")
            st.write("\n".join(status_lines))
            if os.path.exists("chat_history.csv"):
                with open("chat_history.csv","r") as f: st.download_button("📥 Export Chat", f, "chat_history.csv", use_container_width=True)

    # -----------------------------
    # Main Chat
    # -----------------------------
    st.title("💬 Wasla AI Assistant")
    st.markdown("Ask me anything about your documents")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"],1):
                        preview = source[:200]+"..." if len(source)>200 else source
                        st.write(f"**Source {i}:**"); st.write(preview)

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if not db_exists:
                        response="⚠️ **Knowledge base not found.** Please create one first using the button in the sidebar."
                        st.markdown(response); st.session_state.messages.append({"role":"assistant","content":response,"sources":[]}); return
                    if st.session_state.vectorstore is None: st.session_state.vectorstore=init_vectorstore()
                    if st.session_state.vectorstore is None:
                        response="⚠️ **Could not load knowledge base.** Please try recreating it using the button in the sidebar."
                        st.markdown(response); st.session_state.messages.append({"role":"assistant","content":response,"sources":[]}); return
                    if st.session_state.llm is None:
                        with st.spinner("Initializing AI for the first time..."): st.session_state.llm=load_llm()
                    if st.session_state.llm is None:
                        response="❌ Failed to initialize AI. Check your Groq API key."
                        st.markdown(response); st.session_state.messages.append({"role":"assistant","content":response,"sources":[]}); return
                    answer, docs = process_question(prompt, st.session_state.vectorstore, st.session_state.llm)
                    st.markdown(answer)
                    st.session_state.messages.append({"role":"assistant","content":answer,"sources":[doc.page_content[:500] for doc in docs]})
                    save_to_csv(prompt, answer)
                except Exception as e:
                    err_msg=f"❌ Error: {str(e)}"; st.markdown(err_msg); st.session_state.messages.append({"role":"assistant","content":err_msg,"sources":[]})

if __name__=="__main__":
    main()
