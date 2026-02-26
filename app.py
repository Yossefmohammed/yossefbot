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
import random

# Import your constants
try:
    from constant import CHROMA_SETTINGS
except ImportError:
    class CHROMA_SETTINGS:
        persist_directory = "db"

# ===============================
# Response Variation Helpers
# ===============================
class ResponseVariations:
    """Generate varied response openings and patterns"""
    
    GREETINGS = [
        "👋 Hi there!",
        "Hello! 👋", 
        "Great to connect!",
        "Thanks for your question!",
        "I'd be happy to help!",
        "Sure thing!",
        "Happy to assist with that.",
        "Let me look into that for you.",
        "Good question!",
        "I can help with that."
    ]
    
    TRANSITIONS = [
        "Based on what I'm seeing,",
        "According to our documentation,",
        "From what I understand,",
        "Here's what I know:",
        "Let me share what I found:",
        "Looking at our knowledge base,",
        "Here's the information I have:",
        "Great question! Here's what I can tell you:"
    ]
    
    FOLLOW_UPS = [
        "\n\nIs there anything specific about this you'd like to explore further?",
        "\n\nWould you like me to elaborate on any particular aspect?",
        "\n\nFeel free to ask if you want more details about any of these points.",
        "\n\nWhat other questions can I answer for you?",
        "\n\nHappy to dive deeper into any area that interests you.",
        "\n\nIs there something else about Wasla Solutions you'd like to know?"
    ]
    
    UNKNOWN_RESPONSES = [
        "I don't have specific information about that in my knowledge base. However, I can tell you about {topics}. Which would be most helpful?",
        "That's not something I have documented. But I do know about {topics}. Would you like to learn about any of these?",
        "Great question! While I don't have details about that specifically, I can share information about {topics}. What interests you most?",
        "I'm not finding that in my documents. However, my knowledge base includes information about {topics}. Would any of those be helpful?",
        "That's outside what I have in my knowledge base. I can help with {topics} though - just let me know what you'd like to explore."
    ]
    
    @classmethod
    def get_greeting(cls):
        return random.choice(cls.GREETINGS)
    
    @classmethod
    def get_transition(cls):
        return random.choice(cls.TRANSITIONS)
    
    @classmethod
    def get_follow_up(cls):
        return random.choice(cls.FOLLOW_UPS)
    
    @classmethod
    def format_unknown_response(cls, topics):
        topics_str = ", ".join(topics[:3])
        return random.choice(cls.UNKNOWN_RESPONSES).format(topics=topics_str)

# ===============================
# Enhanced Prompt Template
# ===============================
WASLA_PROMPT = PromptTemplate(
    template="""You are Wasla AI, a knowledgeable and friendly assistant for Wasla Solutions.

**CONVERSATION CONTEXT:**
This is message #{msg_count} in our conversation. Previous topics: {previous_topics}

**DOCUMENT CONTEXT:**
{context}

**USER QUESTION:**
{question}

**RESPONSE GUIDELINES:**

1. **Vary Your Openings** - Don't repeat the same phrases:
   - If this is early in conversation: Use a warm greeting
   - If continuing discussion: Reference the previous topic naturally
   - Avoid: "I'd be happy to help with that" (use alternatives)

2. **When You Have Information:**
   - Present key points clearly
   - Use bullet points for lists
   - Be specific - quote or reference document details
   - End with a natural follow-up question

3. **When Information is Missing:**
   - Acknowledge politely
   - Suggest 2-3 related topics you CAN help with
   - Make it conversational, not robotic

4. **Tone & Style:**
   - Professional but warm (like a helpful colleague)
   - Enthusiastic about Wasla's capabilities
   - Natural conversation flow
   - No repetitive patterns

**YOUR RESPONSE:**""",
    input_variables=["context", "question", "msg_count", "previous_topics"]
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
        
        /* Feedback buttons styling */
        div[data-testid="column"] button {
            background-color: #2D3748 !important;
            color: #E5E7EB !important;
            border: 1px solid #4A5568 !important;
            margin: 2px !important;
        }
        div[data-testid="column"] button:hover {
            background-color: #4A5568 !important;
        }
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
# Conversation Tracker
# ===============================
class ConversationTracker:
    """Track conversation context and topics"""
    
    def __init__(self):
        self.topics_discussed = []
        self.message_count = 0
        self.last_response_pattern = None
    
    def add_topic(self, topic):
        if topic and topic not in self.topics_discussed:
            self.topics_discussed.append(topic)
            # Keep only last 5 topics
            if len(self.topics_discussed) > 5:
                self.topics_discussed = self.topics_discussed[-5:]
    
    def increment_count(self):
        self.message_count += 1
    
    def get_context(self):
        return {
            "msg_count": self.message_count,
            "previous_topics": ", ".join(self.topics_discussed) if self.topics_discussed else "none yet"
        }

# ===============================
# System Prompt for LLM (Enhanced with more variety examples)
# ===============================
def get_system_prompt():
    """Return the system prompt with varied response examples"""
    return """You are Wasla AI, a knowledgeable and friendly assistant for Wasla Solutions. 

**Core Personality:**
- Professional, warm, and conversational
- Vary your responses - don't repeat the same patterns
- Be enthusiastic but not overbearing
- Precise and accurate, never inventing information

**Examples of GOOD openings (use these patterns, not exact words):**
- "Thanks for asking about [topic]. Here's what I know..."
- "Great question! Looking at our documentation..."
- "Let me share what I found about [topic]..."
- "Happy to help with that. Based on our knowledge base..."
- "I'd be glad to explain. According to our materials..."
- "That's an excellent question. Let me see what I have on [topic]..."
- "You're asking about [topic] - I can definitely help with that."
- "Here's what I understand from our documents about [topic]..."

**Examples of GOOD follow-ups:**
- "Would you like to know more about [related aspect]?"
- "Is there a specific area of [topic] you're most interested in?"
- "I can also tell you about [related topic 1] or [related topic 2] if helpful."
- "What other questions do you have about Wasla Solutions?"
- "Feel free to ask if you want me to elaborate on any point."
- "Does that answer your question, or would you like more details?"

**Examples of GOOD unknown responses:**
- "I don't have information about that specifically. However, I can tell you about [topic 1], [topic 2], or [topic 3]. Which interests you?"
- "That's not in my knowledge base, but I do know about [topic 1] and [topic 2]. Would either of those be helpful?"
- "Great question! While I can't answer that directly, I can share information about [topic 1], [topic 2], or [topic 3]. What would you like to learn about?"

**BAD patterns to AVOID at all costs:**
- Starting every response with "I'd be happy to help with that"
- Using "Let's dive into" repeatedly
- Copy-pasting the same list of services in every response
- Ending every message the exact same way

**Remember:** The goal is natural, varied conversation that feels helpful, not scripted. Always try to use different opening phrases throughout the conversation.
"""

# ===============================
# Load LLM using Groq (ENHANCED with higher temperature)
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
                            {"role": "system", "content": get_system_prompt()},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.8,  # Increased for more variety
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
# Save to CSV (Chat History)
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
# Save Feedback to CSV
# ===============================
def save_feedback(question, response, feedback):
    """Save user feedback to CSV file"""
    csv_file = "feedback.csv"
    file_exists = os.path.isfile(csv_file)
    
    try:
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Question", "Response", "Feedback", "Time", "Date"])
            writer.writerow([question, response, feedback,
                           datetime.datetime.now().strftime("%H:%M:%S"),
                           datetime.datetime.now().strftime("%Y-%m-%d")])
    except Exception as e:
        pass

# ===============================
# Process Question (ENHANCED with Conversation Tracking)
# ===============================
def process_question(prompt, vectorstore, llm):
    """Process a single question with enhanced context handling and conversation tracking"""
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
        
        # Update conversation tracker
        st.session_state.conversation_tracker.increment_count()
        for topic in topics[:2]:
            st.session_state.conversation_tracker.add_topic(topic)
        
        # Build context with source indication
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Document {i}]: {doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Get conversation context
        conv_context = st.session_state.conversation_tracker.get_context()
        
        # Handle different scenarios
        if len(docs) < 2:  # Few relevant docs found
            # Use unknown response pattern
            if conv_context["msg_count"] > 1:  # Not first message
                # Check if it's a follow-up on previous topic
                if any(topic in prompt.lower() for topic in st.session_state.conversation_tracker.topics_discussed):
                    # It's continuing a previous topic but we lack info
                    follow_up_prompt = f"""The user is asking about {prompt} which relates to previous discussion about {conv_context['previous_topics']}. 
However, I don't have specific information about this in my documents. 
Please acknowledge politely and suggest exploring related aspects we DO have information about from: {', '.join(topics[:3]) if topics else 'services, solutions, or digital transformation'}"""
                else:
                    # New topic with no info
                    follow_up_prompt = prompt
            else:
                follow_up_prompt = prompt
        else:
            follow_up_prompt = prompt
        
        # Format prompt with context
        formatted_prompt = WASLA_PROMPT.format(
            context=context, 
            question=follow_up_prompt,
            msg_count=conv_context["msg_count"],
            previous_topics=conv_context["previous_topics"]
        )
        
        # Get response from LLM
        response = llm.invoke(formatted_prompt)
        
        return response, docs
        
    except Exception as e:
        raise Exception(f"Error processing question: {str(e)}")

# ===============================
# Welcome Message Variations
# ===============================
def get_welcome_message():
    """Return a varied welcome message"""
    welcome_options = [
        """👋 **Hi there! I'm Wasla AI, your guide to all things Wasla Solutions.**

I can help you explore:
- 📋 Our **services and solutions**
- 🔍 **Details** from our knowledge base  
- 💡 **Answers** to your specific questions

**Quick note:** I only share information from our documents, and I'll always be honest if I don't know something. Ready when you are! 🚀""",

        """**Hello! I'm Wasla AI** - think of me as your personal assistant for exploring Wasla Solutions.

Here's what I can do:
- 📋 Share **information** about our services
- 🔍 Help you **navigate** our knowledge base
- 💡 Answer your **specific questions**

**The important stuff:** I stick to what's in our documents, and I'll always show you my sources. What would you like to know?""",

        """**Welcome! I'm Wasla AI**, here to help you discover more about Wasla Solutions.

Need help with:
- 📋 Understanding our **offerings**
- 🔍 Finding **specific information** from documents
- 💡 Getting **answers** to your questions

**Just so you know:** I only provide information from our knowledge base, and I'm always transparent about my sources. Ask me anything! 🤖"""
    ]
    
    return random.choice(welcome_options)

# ===============================
# Main App (ENHANCED with Feedback)
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
    
    if "conversation_tracker" not in st.session_state:
        st.session_state.conversation_tracker = ConversationTracker()
    
    # Add welcome message if not shown
    if "welcome_shown" not in st.session_state:
        st.session_state.welcome_shown = True
        st.session_state.messages.append({
            "role": "assistant", 
            "content": get_welcome_message(),
            "sources": [],
            "feedback": None  # Initialize feedback field
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
                            "sources": [],
                            "feedback": None
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
                # Reset conversation tracker
                st.session_state.conversation_tracker = ConversationTracker()
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
            
            # Export feedback
            if os.path.exists("feedback.csv"):
                with open("feedback.csv", "r") as f:
                    st.download_button("📥 Export Feedback", f, "feedback.csv", use_container_width=True)
    
    # Main chat interface
    st.title("💬 Wasla AI Assistant")
    st.markdown("Ask me anything about your documents")
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for j, source in enumerate(message["sources"], 1):
                        preview = source[:200] + "..." if len(source) > 200 else source
                        st.write(f"**Source {j}:**")
                        st.write(preview)
            
            # Add feedback buttons only for the last assistant message and if not already rated
            if (message["role"] == "assistant" and 
                i == len(st.session_state.messages) - 1 and 
                message.get("feedback") is None):
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Helpful", key=f"fb_pos_{i}"):
                        # Update message feedback
                        st.session_state.messages[i]["feedback"] = "positive"
                        # Get the corresponding user question (previous message)
                        if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                            question = st.session_state.messages[i-1]["content"]
                            save_feedback(question, message["content"], "positive")
                        st.rerun()
                with col2:
                    if st.button("👎 Not helpful", key=f"fb_neg_{i}"):
                        st.session_state.messages[i]["feedback"] = "negative"
                        if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                            question = st.session_state.messages[i-1]["content"]
                            save_feedback(question, message["content"], "negative")
                        st.rerun()
    
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
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response, 
                            "sources": [],
                            "feedback": None
                        })
                        return
                    
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = init_vectorstore()
                    
                    if st.session_state.vectorstore is None:
                        response = "⚠️ **Could not load knowledge base.** Please try recreating it using the button in the sidebar."
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response, 
                            "sources": [],
                            "feedback": None
                        })
                        return
                    
                    if st.session_state.llm is None:
                        with st.spinner("Initializing AI for the first time..."):
                            st.session_state.llm = load_llm()
                    
                    if st.session_state.llm is None:
                        response = "⚠️ **Could not initialize AI.** Please check your Groq API key in the sidebar and click 'Initialize Wasla AI'."
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response, 
                            "sources": [],
                            "feedback": None
                        })
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
                            for j, doc in enumerate(sources, 1):
                                preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                                st.write(f"**Source {j}:**")
                                st.write(preview)
                    
                    # Add to session state with feedback field
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": [doc.page_content for doc in sources] if sources else [],
                        "feedback": None
                    })
                    
                except Exception as e:
                    error_msg = f"❌ **Error:** {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg, 
                        "sources": [],
                        "feedback": None
                    })

if __name__ == "__main__":
    main()