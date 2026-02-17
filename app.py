import streamlit as st
import os
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from ingest import build_vectorstore
from langchain.chat_models import ChatOpenAI
from transformers import pipeline
from langchain.chat_models import HuggingFacePipeline

# =========================
# Constants
# =========================
class CHROMA_SETTINGS:
    persist_directory = "chroma"

WASLA_PROMPT = PromptTemplate(
    template="""
Use the following pieces of context to answer the question.
If you don't know, say that you don't know.

Context: {context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

# =========================
# Styling
# =========================
def set_dark_theme():
    st.markdown("""
    <style>
    .stApp {background-color:#0B1020; color:#EAEAF2;}
    h1 {text-align:center; color:#FFFFFF;}
    textarea {background-color:#111827; color:#E5E7EB;}
    button {background-color:#2563EB !important; color:white !important;}
    footer {visibility:hidden;}
    </style>
    """, unsafe_allow_html=True)

# =========================
# Vector Store
# =========================
@st.cache_resource
def init_vectorstore():
    """Load or build the vector store"""
    persist_dir = CHROMA_SETTINGS.persist_directory
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}
    )

    db_file = os.path.join(persist_dir, "chroma.sqlite3")
    if os.path.exists(db_file):
        return Chroma(embedding_function=embeddings, persist_directory=persist_dir)
    else:
        st.info("⚠️ Vector store not found. Building now...")
        try:
            return build_vectorstore()
        except Exception as e:
            st.error(f"❌ Failed to build vector store: {e}")
            return None

# =========================
# Load OpenAI LLM
# =========================
@st.cache_resource
def load_llm():
    """Load a free HuggingFace model"""
    try:
        # Use a local pipeline (or HF Inference API)
        hf_pipeline = pipeline(
            "text-generation",
            model="meta-llama/Llama-2-7b-chat-hf",  # can be replaced with smaller free models
            max_new_tokens=500,
            temperature=0.3,
            device_map="auto"  # uses GPU if available
        )
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        return llm
    except Exception as e:
        st.error(f"❌ Failed to load HuggingFace LLM: {e}")
        return None

# =========================
# Main app
# =========================
def main():
    st.set_page_config(page_title="Wasla Chatbot", page_icon="🤖", layout="wide")
    set_dark_theme()

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vectorstore" not in st.session_state:
        with st.spinner("Loading knowledge base..."):
            st.session_state.vectorstore = init_vectorstore()
    
    if "llm" not in st.session_state:
        with st.spinner("Loading language model..."):
            st.session_state.llm = load_llm()
    
    if not st.session_state.llm:
        st.warning("⚠️ LLM not loaded. Please check your API key.")
        return
    if not st.session_state.vectorstore:
        st.warning("⚠️ Vector store not loaded. Please check your setup.")
        return

    st.title("💬 Wasla Solutions Chatbot")
    st.markdown("Ask questions about your PDF documents")

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=st.session_state.llm,
                        chain_type="stuff",
                        retriever=retriever,
                        chain_type_kwargs={"prompt": WASLA_PROMPT},
                        return_source_documents=True
                    )
                    result = qa_chain.invoke({"query": prompt})
                    response = result.get("result", "⚠️ No answer returned")

                    # Optional: show sources
                    sources = result.get("source_documents", [])
                    if sources:
                        response += "\n\n**Sources:**\n" + "\n".join(
                            [doc.metadata.get("source", "unknown") for doc in sources]
                        )

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": str(e)})

if __name__ == "__main__":
    main()
