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

# ===============================
# Constants / Settings
# ===============================
try:
    from constant import CHROMA_SETTINGS
except ImportError:
    class CHROMA_SETTINGS:
        persist_directory = "db"

# ===============================
# Enhanced Prompt Template
# ===============================
WASLA_PROMPT = PromptTemplate(
    template="""
You are a knowledgeable assistant for Wasla Solutions. Your role is to provide helpful, accurate, and engaging responses based strictly on the provided context.

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Start your response by restating the user's question.
2. Answer based ONLY on the context provided above.
3. If the context doesn't contain the answer, say: "I don't have information about that in my knowledge base. Could you ask something else about Wasla Solutions?"
4. Be conversational and professional.
5. Keep responses concise but informative.
6. Use bullet points for lists when helpful.
7. End by offering additional help.

YOUR RESPONSE:
""",
    input_variables=["context", "question"]
)

# ===============================
# Dark Theme
# ===============================
def set_dark_theme():
    st.markdown(
        """
        <style>
        .stApp { background-color: #0B1020; color: #EAEAF2; }
        section.main > div { max-width: 900px; margin: auto; }
        h1 { text-align: center; font-size: 42px; font-weight: 700; color: #FFFFFF; margin-bottom: 0px; }
        .stChatMessage { background-color: #1E1E2E; border-radius: 10px; padding: 10px; margin: 5px 0; border: 1px solid #2D3748; }
        [data-testid="chatMessageContent"] { color: #E5E7EB !important; }
        textarea { background-color: #111827 !important; color: #E5E7EB !important; border-radius: 10px !important; border: 1px solid #2D3748 !important; }
        button { background-color: #2563EB !important; color: white !important; border-radius: 10px !important; width: 100%; transition: all 0.3s ease; }
        button:hover { background-color: #1D4ED8 !important; box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3); }
        .stExpander { background-color: #1E1E2E; border: 1px solid #2D3748; border-radius: 10px; }
        .stSpinner > div { border-color: #2563EB !important; }
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
            "services","solutions","products","digital","platform",
            "fintech","e-commerce","marketing","sales","software",
            "development","mobile","web","ui/ux","branding",
            "consulting","strategy","innovation","technology"
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
                        messages=[
                            {"role":"system","content":"You are Wasla AI, a knowledgeable assistant. Respond professionally, be precise, vary greetings, show sources if available."},
                            {"role":"user","content":prompt}
                        ],
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
                if test_response and "Error" not in test_response:
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
        if not docs_path.exists():
            docs_path.mkdir(exist_ok=True)
            status_text.text("📁 Created 'docs' folder. Please add PDFs."); progress_bar.progress(100); time.sleep(2); return False
        pdf_files = list(docs_path.glob("**/*.pdf"))
        if not pdf_files: status_text.text("❌ No PDF files found."); progress_bar.progress(100); time.sleep(2); return False

        all_documents = []
        for i, pdf_path in enumerate(pdf_files):
            status_text.text(f"📄 Loading: {pdf_path.name} ({i+1}/{len(pdf_files)})")
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            all_documents.extend(documents)
            progress_bar.progress(20 + int(30*(i+1)/len(pdf_files)))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(all_documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device":"cpu"})
        db_path = Path("db"); db_path.mkdir(exist_ok=True)
        db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=str(db_path))
        db.persist()
        status_text.text("✅ Database created successfully!"); progress_bar.progress(100); time.sleep(2)
        status_text.empty(); progress_bar.empty()
        return True
    except Exception as e:
        status_text.text(f"❌ Error: {str(e)}"); progress_bar.progress(100); time.sleep(2); status_text.empty(); progress_bar.empty(); return False

# ===============================
# Initialize Vector Store
# ===============================
@st.cache_resource(ttl=3600)
def init_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device":"cpu"})
        persist_dir = "db"
        if os.path.exists(os.path.join(persist_dir,"chroma.sqlite3")):
            db = Chroma(embedding_function=embeddings, persist_directory=persist_dir)
            try: count = db._collection.count(); st.sidebar.success(f"✅ Database loaded with {count} documents")
            except: st.sidebar.warning("⚠️ Database loaded but may be empty")
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
    csv_file = "chat_history.csv"
    file_exists = os.path.isfile(csv_file)
    try:
        with open(csv_file,"a",newline="",encoding="utf-8") as f:
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
            enhanced_prompt = f"{prompt}\n\n(Note: Relevant topics: {topic_str})"
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

    # Welcome message
    if "welcome_shown" not in st.session_state:
        st.session_state.welcome_shown=True
        st.session_state.messages.append({"role":"assistant","content":"👋 **Hello! I'm Wasla AI, your intelligent assistant.**\nI can answer questions from your documents or knowledge base."})

    # -----------------------------
    # Sidebar
    # -----------------------------
    with st.sidebar:
        st.title("🤖 Wasla Solutions")
        st.markdown("---")
        st.subheader("🔑 API Configuration")
        if "GROQ_API_KEY" in st.secrets:
            st.success("✅ Groq API key configured")
            if st.button("🔄 Initialize Wasla AI"):
                st.session_state.llm = load_llm()
                if st.session_state.llm: st.success("✅ Wasla AI ready!")
                else: st.error("❌ Failed to initialize AI")
        else:
            st.error("❌ GROQ_API_KEY not found")

        st.markdown("---")
        st.subheader("📚 Knowledge Base")
        db_path = Path("db"); db_exists = (db_path / "chroma.sqlite3").exists()
        if db_exists:
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = init_vectorstore()
            else: st.success("✅ Knowledge base ready")
        else:
            docs_path = Path("docs"); pdf_files = list(docs_path.glob("**/*.pdf")) if docs_path.exists() else []
            if pdf_files:
                if st.button("🚀 Create Knowledge Base"):
                    create_database_from_pdfs()
            else:
                st.info("Please add PDFs to the 'docs' folder")

        st.markdown("---")
        st.subheader("🎮 Controls")
        col1,col2=st.columns(2)
        with col1:
            if st.button("🔄 Reset AI"):
                st.cache_resource.clear(); st.session_state.llm=None; st.success("✅ AI reset"); st.rerun()
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages=[st.session_state.messages[0]] if st.session_state.messages else []; st.rerun()

    # -----------------------------
    # Main Chat
    # -----------------------------
    st.title("💬 Wasla AI Assistant")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("assistant"):
            try:
                if st.session_state.vectorstore is None:
                    st.session_state.vectorstore = init_vectorstore()
                if st.session_state.vectorstore is None:
                    response = "⚠️ Knowledge base missing."
                    st.markdown(response); st.session_state.messages.append({"role":"assistant","content":response,"sources":[]}); return
                if st.session_state.llm is None:
                    st.session_state.llm = load_llm()
                answer, docs = process_question(prompt, st.session_state.vectorstore, st.session_state.llm)
                # Combine question + answer for display
                full_answer = f"**Q:** {prompt}\n\n**A:** {answer}"
                st.markdown(full_answer)
                st.session_state.messages.append({
                    "role":"assistant",
                    "content":full_answer,
                    "sources":[doc.page_content[:500] for doc in docs]
                })
                save_to_csv(prompt, answer)
            except Exception as e:
                err_msg=f"❌ Error: {str(e)}"; st.markdown(err_msg); st.session_state.messages.append({"role":"assistant","content":err_msg,"sources":[]})

if __name__=="__main__":
    main()