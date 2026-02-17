import streamlit as st
import os
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from ingest import build_vectorstore

# Constants
class CHROMA_SETTINGS:
    persist_directory = "chroma"

# Prompt template
WASLA_PROMPT = PromptTemplate(
    template="""
Use the following pieces of context to answer the question. 
If you don't know, say that you don't know.

Context: {context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

# Dark theme
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

# Initialize vector store
@st.cache_resource
def init_vectorstore():
    """Load or build the vector store"""
    persist_dir = CHROMA_SETTINGS.persist_directory
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}
    )

    # If DB exists, load it
    if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
        db = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        return db
    else:
        st.info("⚠️ Vector store not found. Building now...")
        db = build_vectorstore()
        return db

# Main app
def main():
    st.set_page_config(page_title="Wasla Chatbot", page_icon="🤖", layout="wide")
    set_dark_theme()

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vectorstore" not in st.session_state:
        with st.spinner("Loading knowledge base..."):
            st.session_state.vectorstore = init_vectorstore()

    st.title("💬 Wasla Solutions Chatbot")
    st.markdown("Ask questions about your PDF documents")

    # Chat display
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if not st.session_state.vectorstore:
                        response = "⚠️ Knowledge base not loaded."
                    else:
                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k":3})
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=None,  # Replace with your LLM
                            chain_type="stuff",
                            retriever=retriever,
                            chain_type_kwargs={"prompt": WASLA_PROMPT},
                            return_source_documents=True
                        )
                        result = qa_chain.invoke({"query": prompt})
                        response = result['result']

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": str(e)})

if __name__ == "__main__":
    main()
