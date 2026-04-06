import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import csv
from pathlib import Path
import time
import datetime
import re
import random

# ===============================
# ✅ CONFIGURATION — EDIT HERE
# ===============================
CONFIG = {
    "bot_name": "AI Assistant",            # Change to your bot's name
    "company_name": "Wasla",        # Change to your company name
    "page_title": "AI Assistant Chatbot",  # Browser tab title
    "page_icon": "🤖",                     # Browser tab icon
    "primary_color": "#2563EB",            # Main accent color (hex)
    "topic_keywords": [                    # Keywords relevant to YOUR documents
        "services", "products", "solutions", "support",
        "pricing", "features", "team", "contact",
        "policy", "process", "guide", "help"
    ],
    "groq_models": [                       # Models to try in order
        "llama-3.3-70b-versatile",
        "deepseek-r1-distill-llama-70b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "gemma2-9b-it"
    ]
}

# ===============================
# Greeting Detection
# ===============================
def is_greeting(query):
    patterns = [
        r'^\s*(hi|hello|hey|greetings|howdy|sup|yo|hiya|heya)\s*[!.]?\s*$',
        r'^(hi|hello|hey)\s+(there|chat|bot|assistant)\s*[!.]?\s*$',
        r'how are you',
        r"how's it going",
        r"what's up",
        r'good (morning|afternoon|evening)',
        r'nice to meet you',
    ]
    query_lower = query.lower().strip()
    for pattern in patterns:
        if re.search(pattern, query_lower):
            return True
    return False

def get_greeting_response():
    bot = CONFIG["bot_name"]
    company = CONFIG["company_name"]
    responses = [
        f"Hi there! 👋 I'm {bot}, your assistant for {company}. What can I help you with today?",
        f"Hello! Great to connect. I'm {bot} — feel free to ask me anything about {company}.",
        f"Hey! 👋 I'm here and ready to help. What would you like to know about {company}?",
        f"Hi! I'm {bot}. Ask me anything about {company}'s services, solutions, or anything else on your mind.",
        f"Hello! What can I do for you today? I'm happy to help with any questions about {company}.",
    ]
    return random.choice(responses)

# ===============================
# System Prompt
# ===============================
def get_system_prompt():
    bot = CONFIG["bot_name"]
    company = CONFIG["company_name"]
    topics = ", ".join(CONFIG["topic_keywords"][:6])

    return f"""You are {bot}, a professional and friendly AI assistant for {company}.

## YOUR ROLE
Answer questions based on the provided document context. Be helpful, accurate, and conversational.

## CRITICAL RULES — FOLLOW STRICTLY
1. NEVER refuse a question about {company} — always try to help
2. NEVER say a request is unethical or inappropriate unless it clearly is harmful
3. NEVER invent facts, numbers, or details not in the documents
4. If you don't have the answer, say so clearly and suggest related topics
5. NEVER give vague, empty answers just to fill space
6. Keep responses focused and useful — no unnecessary padding

## RESPONSE VARIETY — STRICTLY ENFORCED
You are FORBIDDEN from using the same opening phrase twice in a row.
NEVER start with "Let me share what I know", "Let's dive into", or "Great to have you here" more than once.

Rotate between these response styles:
- Direct: "{company} handles this by..."
- Fact-first: "Based on our documents, here's what I know..."
- Acknowledge: "Good question. Here's what we offer..."
- Contextual: "Since you're asking about [topic], here's the relevant info..."
- Concise: Just answer directly without any preamble

## WHEN YOU DON'T HAVE INFORMATION
- Say clearly: "I don't have specific details on that in my knowledge base."
- Suggest 2-3 related topics you DO know about from: {topics}
- Never make up information to fill the gap

## WHEN ASKED TO INTRODUCE YOURSELF
Give a warm, confident 3-4 sentence introduction that covers:
1. Your name and role as AI assistant for {company}
2. What {company} does (based on documents)
3. The key areas you can help the user with
4. An invitation to ask questions
Never give a one-liner introduction — make it welcoming and informative.

## WHEN THE USER SAYS THEY DON'T UNDERSTAND
If the user says things like "I don't understand", "explain more simply", "I can't understand", "please clarify":
- Start with something warm like "Let me put it more simply:" or "No problem, let me explain differently:"
- Use plain, everyday language — avoid jargon
- Use a real-world analogy if it helps
- Break it into very small, clear steps
- Be patient and encouraging in tone

## HANDLE TYPOS & MISSPELLINGS
Always try to understand the customer's intent even with spelling mistakes.
Common examples: "noline"="online", "websit"="website", "servise"="service", "cantact"="contact"
Never ask the customer to repeat because of a typo. Just understand and respond naturally.

## LEAD COLLECTION
When a customer shows clear interest in a specific service or asks about building something:
- Naturally ask for their name ONCE early in that conversation
- Example: "I'd love to help with that! May I ask your name so I can assist you better?"
- Once you have their name, USE IT in follow-up responses to make it personal
- Never ask for the name more than once

## RESPONSE LENGTH
- Keep responses SHORT — 3 to 5 sentences max for simple questions
- Only use bullet points when listing 3 or more distinct items
- Don't repeat information already covered in the conversation
- Answer the question first, then offer to elaborate if needed

## PROACTIVE CALL TO ACTION
After 3 or more exchanges about a specific customer need, PROACTIVELY suggest next steps.
Don't wait for the customer to ask. Say something like:
"It sounds like you have a clear vision! Would you like to book a free consultation to discuss this in detail?
You can reach us at info@waslasolutions.com or book via our Calendly link on the Wasla website."

## TONE & STYLE
- Professional but warm — like a knowledgeable colleague
- Be direct and confident, not vague or uncertain
- Use bullet points for lists, keep prose for simple answers
- SHORT and focused — quality over quantity
"""

# ===============================
# Prompt Template with History
# ===============================
BOT_PROMPT = PromptTemplate(
    template="""You are {bot_name}, assistant for {company_name}.

## CONVERSATION HISTORY (last few exchanges):
{history}

## RELEVANT DOCUMENTS:
{context}

## CURRENT QUESTION:
{question}

## INSTRUCTIONS:
- Answer based on the document context above
- Use the conversation history to understand context and avoid repeating yourself
- If documents don't contain the answer, say so clearly and suggest related topics
- Vary your response style — do NOT repeat the same opening as your last response
- Be concise, accurate, and helpful

YOUR RESPONSE:""",
    input_variables=["bot_name", "company_name", "history", "context", "question"]
)

# ===============================
# UI Theme
# ===============================
def set_theme():
    color = CONFIG["primary_color"]
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        * {{ font-family: 'Inter', sans-serif; }}
        .stApp {{
            background-color: #0B1020;
            color: #EAEAF2;
        }}
        section.main > div {{
            max-width: 900px;
            margin: auto;
        }}
        h1 {{
            text-align: center;
            font-size: 36px;
            font-weight: 700;
            color: #FFFFFF;
        }}
        .stChatMessage {{
            background-color: #1E1E2E;
            border-radius: 12px;
            padding: 12px;
            margin: 6px 0;
            border: 1px solid #2D3748;
        }}
        [data-testid="chatMessageContent"] {{
            color: #E5E7EB !important;
        }}
        textarea {{
            background-color: #111827 !important;
            color: #E5E7EB !important;
            border-radius: 10px !important;
            border: 1px solid #2D3748 !important;
        }}
        .stButton > button {{
            background-color: {color} !important;
            color: white !important;
            border-radius: 10px !important;
            border: none !important;
            transition: all 0.2s ease;
        }}
        .stButton > button:hover {{
            opacity: 0.85 !important;
        }}
        .stExpander {{
            background-color: #1E1E2E;
            border: 1px solid #2D3748;
            border-radius: 10px;
        }}
        footer {{visibility: hidden;}}
        div[data-testid="column"] button {{
            background-color: #2D3748 !important;
            color: #E5E7EB !important;
            border: 1px solid #4A5568 !important;
        }}
        </style>
    """, unsafe_allow_html=True)

# ===============================
# Topic Extraction
# ===============================
def extract_topics_from_docs(docs, max_topics=3):
    try:
        if len(docs) < 1:
            return []
        all_text = " ".join([doc.page_content.lower() for doc in docs[:3]])
        found = [kw for kw in CONFIG["topic_keywords"] if kw in all_text]
        return list(set(found))[:max_topics]
    except:
        return []

# ===============================
# Conversation Tracker
# ===============================
class ConversationTracker:
    def __init__(self):
        self.topics_discussed = []
        self.message_count = 0
        self.last_opening = ""

    def add_topic(self, topic):
        if topic and topic not in self.topics_discussed:
            self.topics_discussed.append(topic)
            if len(self.topics_discussed) > 5:
                self.topics_discussed = self.topics_discussed[-5:]

    def increment_count(self):
        self.message_count += 1

    def set_last_opening(self, response_text):
        # Store first 6 words of response to track repetition
        words = response_text.strip().split()[:6]
        self.last_opening = " ".join(words)

    def get_context(self):
        return {
            "msg_count": self.message_count,
            "previous_topics": ", ".join(self.topics_discussed) if self.topics_discussed else "none yet",
            "last_opening": self.last_opening
        }

# ===============================
# Build Conversation History String
# ===============================
def build_history_string(messages, max_exchanges=4):
    """Take last N user/assistant exchanges and format as readable history."""
    history_lines = []
    # Get last max_exchanges*2 messages (each exchange = 1 user + 1 assistant)
    recent = [m for m in messages if m["role"] in ("user", "assistant")][-max_exchanges * 2:]
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Truncate long messages in history to keep prompt manageable
        content = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
        history_lines.append(f"{role}: {content}")
    return "\n".join(history_lines) if history_lines else "No previous conversation."

# ===============================
# Load LLM via Groq
# ===============================
@st.cache_resource(ttl=3600)
def load_llm():
    try:
        from groq import Groq

        if "GROQ_API_KEY" not in st.secrets:
            st.sidebar.error("❌ GROQ_API_KEY not found in secrets!")
            return None

        client = Groq(api_key=st.secrets["GROQ_API_KEY"])

        class GroqLLM:
            def __init__(self, client, model_name):
                self.client = client
                self.model = model_name

            def invoke(self, prompt):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": get_system_prompt()},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1500,   # FIX: increased from 600 to prevent cut-offs
                        top_p=0.9
                    )
                    return completion.choices[0].message.content
                except Exception as e:
                    return f"Sorry, I ran into an error: {str(e)}. Please try again."

        placeholder = st.sidebar.empty()
        placeholder.info("🔄 Initializing AI...")

        for model_name in CONFIG["groq_models"]:
            try:
                placeholder.info(f"🔄 Testing {model_name}...")
                llm = GroqLLM(client, model_name)
                test = llm.invoke("Say 'ready' in one word.")
                if test and "Error" not in test:
                    short_name = model_name.split("/")[-1]
                    placeholder.success(f"✅ AI ready ({short_name})")
                    return llm
            except:
                continue

        placeholder.error("❌ Could not load any AI model.")
        return None

    except ImportError:
        st.sidebar.error("❌ Install groq: pip install groq")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
        return None

# ===============================
# Build Vector Database from PDFs
# ===============================
def create_database_from_pdfs():
    progress = st.progress(0)
    status = st.empty()

    try:
        status.text("📁 Checking for PDF files...")
        progress.progress(10)

        docs_path = Path("docs")
        docs_path.mkdir(exist_ok=True)

        pdf_files = list(docs_path.glob("**/*.pdf"))
        if not pdf_files:
            status.text("❌ No PDFs found in 'docs/' folder.")
            progress.progress(100)
            time.sleep(2)
            status.empty(); progress.empty()
            return False

        status.text(f"📚 Found {len(pdf_files)} PDF(s). Loading...")
        progress.progress(20)

        all_docs = []
        for i, pdf in enumerate(pdf_files):
            status.text(f"📄 Loading: {pdf.name} ({i+1}/{len(pdf_files)})")
            loader = PyPDFLoader(str(pdf))
            all_docs.extend(loader.load())
            progress.progress(20 + int(25 * (i + 1) / len(pdf_files)))

        status.text(f"✅ Loaded {len(all_docs)} pages. Splitting into chunks...")
        progress.progress(50)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(all_docs)

        status.text(f"🔤 Creating embeddings for {len(chunks)} chunks...")
        progress.progress(65)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"}
        )

        db_path = Path("db")
        db_path.mkdir(exist_ok=True)

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(db_path)
        )
        db.persist()

        progress.progress(100)
        status.text("✅ Knowledge base created successfully!")
        time.sleep(2)
        status.empty(); progress.empty()
        return True

    except Exception as e:
        status.text(f"❌ Error: {str(e)}")
        time.sleep(3)
        status.empty(); progress.empty()
        return False

# ===============================
# Load Existing Vector Store
# ===============================
@st.cache_resource(ttl=3600)
def init_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"}
        )
        db_file = os.path.join("db", "chroma.sqlite3")
        if os.path.exists(db_file):
            db = Chroma(embedding_function=embeddings, persist_directory="db")
            try:
                count = db._collection.count()
                st.sidebar.success(f"✅ Loaded {count} document chunks")
            except:
                st.sidebar.warning("⚠️ Database loaded but may be empty")
            return db
        else:
            st.sidebar.warning("📁 No knowledge base found.")
            return None
    except Exception as e:
        st.sidebar.error(f"❌ Vector store error: {str(e)}")
        return None

# ===============================
# Save Chat History & Feedback
# ===============================
def save_to_csv(question, answer):
    csv_file = "chat_history.csv"
    file_exists = os.path.isfile(csv_file)
    try:
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Question", "Answer", "Time", "Date"])
            writer.writerow([
                question, answer,
                datetime.datetime.now().strftime("%H:%M:%S"),
                datetime.datetime.now().strftime("%Y-%m-%d")
            ])
    except:
        pass

def save_feedback(question, response, feedback):
    csv_file = "feedback.csv"
    file_exists = os.path.isfile(csv_file)
    try:
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Question", "Response", "Feedback", "Time", "Date"])
            writer.writerow([
                question, response, feedback,
                datetime.datetime.now().strftime("%H:%M:%S"),
                datetime.datetime.now().strftime("%Y-%m-%d")
            ])
    except:
        pass

# ===============================
# Process Question — IMPROVED
# ===============================
def process_question(prompt, vectorstore, llm):
    try:
        # FIX 1: Handle greetings at ANY point in conversation (removed <= 2 limit)
        if is_greeting(prompt):
            st.session_state.tracker.increment_count()
            return get_greeting_response(), []

        # Retrieve relevant documents
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 15}
        )
        docs = retriever.get_relevant_documents(prompt)

        # FIX 3: Filter out empty or whitespace-only document chunks
        docs = [doc for doc in docs if doc.page_content.strip()]

        # FIX 2: Relevance check — if no docs found, return clean fallback
        if len(docs) == 0:
            st.session_state.tracker.increment_count()
            topics = ", ".join(CONFIG["topic_keywords"][:4])
            return (
                f"I don't have specific information about that in my knowledge base. "
                f"I can help you with topics like **{topics}**. What would you like to know?",
                []
            )

        # Update tracker
        topics = extract_topics_from_docs(docs)
        st.session_state.tracker.increment_count()
        for topic in topics[:2]:
            st.session_state.tracker.add_topic(topic)

        # Build context string from retrieved docs
        context = "\n\n".join([
            f"[Document {i+1}]:\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])

        # FIX 3: Build real conversation history for multi-turn memory
        history = build_history_string(st.session_state.messages, max_exchanges=4)

        # Format prompt
        formatted_prompt = BOT_PROMPT.format(
            bot_name=CONFIG["bot_name"],
            company_name=CONFIG["company_name"],
            history=history,
            context=context,
            question=prompt
        )

        # Get response
        response = llm.invoke(formatted_prompt)

        # Track last opening to help enforce variety
        st.session_state.tracker.set_last_opening(response)

        return response, docs

    except Exception as e:
        raise Exception(f"Error processing question: {str(e)}")

# ===============================
# Welcome Message
# ===============================
def get_welcome_message():
    bot = CONFIG["bot_name"]
    company = CONFIG["company_name"]
    options = [
        f"""👋 **Hi! I'm {bot}, your AI assistant for {company}.**

I can help you with:
- 📋 Questions about our **services and products**
- 🔍 Finding **specific information** from our documents
- 💡 Getting **clear, accurate answers** fast

I only answer from our knowledge base, so my answers are always grounded in real information. What would you like to know?""",

        f"""**Hello! I'm {bot}** — here to help you explore {company}.

Ask me about:
- 📋 Our **offerings and capabilities**
- 🔍 **Specific details** from our documents
- 💡 How we can **support your needs**

What's on your mind?""",
    ]
    return random.choice(options)

# ===============================
# Main App
# ===============================
def main():
    st.set_page_config(
        page_title=CONFIG["page_title"],
        page_icon=CONFIG["page_icon"],
        layout="wide",
        initial_sidebar_state="expanded"
    )

    set_theme()

    # Session state initialization
    defaults = {
        "messages": [],
        "llm": None,
        "vectorstore": None,
        "tracker": ConversationTracker(),
        "welcome_shown": False,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Show welcome message once
    if not st.session_state.welcome_shown:
        st.session_state.welcome_shown = True
        st.session_state.messages.append({
            "role": "assistant",
            "content": get_welcome_message(),
            "sources": [],
            "feedback": None
        })

    db_exists = Path("db/chroma.sqlite3").exists()

    # ---- SIDEBAR ----
    with st.sidebar:
        st.title(f"{CONFIG['page_icon']} {CONFIG['company_name']}")
        st.markdown("---")

        # API Key
        st.subheader("🔑 API Setup")
        if "GROQ_API_KEY" in st.secrets:
            st.success("✅ Groq API key found")
            if st.button("🚀 Initialize AI", use_container_width=True):
                with st.spinner("Loading AI model..."):
                    st.session_state.llm = load_llm()
                if st.session_state.llm:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"✅ **{CONFIG['bot_name']} is ready!** Ask me anything about {CONFIG['company_name']}.",
                        "sources": [],
                        "feedback": None
                    })
        else:
            st.error("❌ GROQ_API_KEY not set")
            st.info(
                "Add your key to `.streamlit/secrets.toml`:\n"
                "```\nGROQ_API_KEY = 'your_key_here'\n```\n"
                "Get a free key at [console.groq.com](https://console.groq.com)"
            )

        st.markdown("---")

        # Knowledge Base
        st.subheader("📚 Knowledge Base")
        if db_exists:
            if st.session_state.vectorstore is None:
                with st.spinner("Loading knowledge base..."):
                    st.session_state.vectorstore = init_vectorstore()
            else:
                st.success("✅ Knowledge base ready")
            if st.session_state.vectorstore:
                try:
                    count = st.session_state.vectorstore._collection.count()
                    st.info(f"📊 {count} document chunks loaded")
                except:
                    pass
        else:
            st.warning("No knowledge base found")
            docs_path = Path("docs")
            pdf_files = list(docs_path.glob("**/*.pdf")) if docs_path.exists() else []
            if pdf_files:
                st.info(f"📄 {len(pdf_files)} PDF(s) ready to process")
                if st.button("⚙️ Build Knowledge Base", type="primary", use_container_width=True):
                    success = create_database_from_pdfs()
                    if success:
                        st.success("✅ Done!")
                        st.rerun()
            else:
                st.info("Add PDF files to the `docs/` folder, then come back here.")

        st.markdown("---")

        # Controls
        st.subheader("⚙️ Controls")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset AI", use_container_width=True):
                st.cache_resource.clear()
                st.session_state.llm = None
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                first_msg = st.session_state.messages[0] if st.session_state.messages else None
                st.session_state.messages = [first_msg] if first_msg else []
                st.session_state.tracker = ConversationTracker()
                st.rerun()

        # Status & Exports
        with st.expander("ℹ️ System Status"):
            st.write(f"🔑 API Key: {'✅' if 'GROQ_API_KEY' in st.secrets else '❌'}")
            st.write(f"🤖 AI Model: {'✅' if st.session_state.llm else '❌'}")
            st.write(f"📚 Knowledge Base: {'✅' if db_exists else '❌'}")
            st.write(f"💬 Messages: {len(st.session_state.messages)}")
            st.write(f"🧠 Topics tracked: {', '.join(st.session_state.tracker.topics_discussed) or 'none yet'}")

            if os.path.exists("chat_history.csv"):
                with open("chat_history.csv", "r") as f:
                    st.download_button("📥 Export Chat", f, "chat_history.csv", use_container_width=True)
            if os.path.exists("feedback.csv"):
                with open("feedback.csv", "r") as f:
                    st.download_button("📥 Export Feedback", f, "feedback.csv", use_container_width=True)

    # ---- MAIN CHAT UI ----
    st.title(f"💬 {CONFIG['bot_name']}")
    st.markdown(f"*Powered by {CONFIG['company_name']}'s knowledge base*")

    # Display messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if available — filter empty ones
            if message.get("sources"):
                clean_sources = [s for s in message["sources"] if s.strip()]
                if clean_sources:
                    with st.expander(f"📚 View Sources ({len(clean_sources)})"):
                        for j, source in enumerate(clean_sources, 1):
                            preview = source[:250] + "..." if len(source) > 250 else source
                            st.write(f"**Source {j}:** {preview}")

            # Feedback buttons on last assistant message only
            if (message["role"] == "assistant"
                    and i == len(st.session_state.messages) - 1
                    and message.get("feedback") is None
                    and i > 0):  # don't show on welcome message
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Helpful", key=f"pos_{i}"):
                        st.session_state.messages[i]["feedback"] = "positive"
                        if st.session_state.messages[i - 1]["role"] == "user":
                            save_feedback(
                                st.session_state.messages[i - 1]["content"],
                                message["content"], "positive"
                            )
                        st.rerun()
                with col2:
                    if st.button("👎 Not helpful", key=f"neg_{i}"):
                        st.session_state.messages[i]["feedback"] = "negative"
                        if st.session_state.messages[i - 1]["role"] == "user":
                            save_feedback(
                                st.session_state.messages[i - 1]["content"],
                                message["content"], "negative"
                            )
                        st.rerun()

    # ---- CHAT INPUT ----
    if prompt := st.chat_input(f"Ask {CONFIG['bot_name']} anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Check: knowledge base exists
                    if not db_exists:
                        response = (
                            "⚠️ **No knowledge base found.** "
                            "Please add PDFs to the `docs/` folder and build the knowledge base from the sidebar."
                        )
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant", "content": response,
                            "sources": [], "feedback": None
                        })
                        return

                    # Load vectorstore if needed
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = init_vectorstore()

                    if st.session_state.vectorstore is None:
                        response = (
                            "⚠️ **Could not load knowledge base.** "
                            "Try rebuilding it from the sidebar."
                        )
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant", "content": response,
                            "sources": [], "feedback": None
                        })
                        return

                    # Load LLM if needed
                    if st.session_state.llm is None:
                        with st.spinner("Starting AI for the first time..."):
                            st.session_state.llm = load_llm()

                    if st.session_state.llm is None:
                        response = (
                            "⚠️ **AI could not start.** "
                            "Please check your Groq API key in the sidebar."
                        )
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant", "content": response,
                            "sources": [], "feedback": None
                        })
                        return

                    # Process the question
                    response, sources = process_question(
                        prompt,
                        st.session_state.vectorstore,
                        st.session_state.llm
                    )

                    # Save to history
                    save_to_csv(prompt, response)

                    # Display response
                    st.markdown(response)

                    # Show sources — filter empty ones
                    if sources:
                        clean_sources = [doc for doc in sources if doc.page_content.strip()]
                        if clean_sources:
                            with st.expander(f"📚 View Sources ({len(clean_sources)})"):
                                for j, doc in enumerate(clean_sources, 1):
                                    preview = (
                                        doc.page_content[:250] + "..."
                                        if len(doc.page_content) > 250
                                        else doc.page_content
                                    )
                                    st.write(f"**Source {j}:** {preview}")

                    # Store message — filter empty sources
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": [doc.page_content for doc in sources if doc.page_content.strip()] if sources else [],
                        "feedback": None
                    })

                except Exception as e:
                    error_msg = f"❌ **Error:** {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", "content": error_msg,
                        "sources": [], "feedback": None
                    })


if __name__ == "__main__":
    main()