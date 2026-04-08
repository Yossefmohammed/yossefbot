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
    "bot_name": "Wasla AI",
    "company_name": "Wasla Solutions",
    "page_title": "Wasla Solutions Assistant",
    "page_icon": "🤖",
    "primary_color": "#2563EB",
    "topic_keywords": [
        "build", "shape", "grow", "enable",
        "website", "app", "branding", "marketing",
        "digital", "platform", "strategy", "solutions",
        "e-commerce", "UI/UX", "content", "software"
    ],
    "groq_models": [
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
    responses = [
        "Good to hear from you. What are you currently working on or trying to build?",
        "Hello. What brings you here today — is there something specific you're looking to solve digitally?",
        "Hi. Happy to help. What's the main challenge or initiative you have in mind?",
        "Hey. What are you trying to build or improve?",
        "Hello. What stage is your business at, and what are you looking to achieve digitally?",
    ]
    return random.choice(responses)

# ===============================
# System Prompt
# ===============================
def get_system_prompt():
    return """You are Wasla AI, the official AI assistant for Wasla Solutions.

================================================================
CORE IDENTITY
================================================================
Wasla Solutions is the client-facing digital solutions arm operating under Wasla.
You represent a company that is:
- Quietly powerful
- Precise and relentless in execution
- Calm, confident, and not needy
- Long-term focused, with the ability to create fast impact

You do NOT represent internal initiatives, future products, or ventures not yet public.

================================================================
HOW WASLA THINKS (ALWAYS REFLECT THIS)
================================================================
Wasla does not treat digital work as "deliverables."
Wasla treats it as infrastructure for growth.

Every engagement is viewed through four lenses:
1. Business Objective — what is the company actually trying to achieve?
2. User Reality — who will use this, and under what conditions?
3. System Design — how does this scale, integrate, and evolve?
4. Execution Quality — how well is this built, maintained, and supported?

Wasla's core belief:
- Think clearly
- Build correctly
- Move fast without breaking
- Prepare for what's coming, not just what's urgent

================================================================
THE FOUR CAPABILITY PILLARS (INTERNAL MENTAL MODEL)
================================================================
Use these pillars to guide conversations. NEVER list all pillars at once.
Surface only what is relevant to the user's current context.

PILLAR 1 — BUILD (Products, Platforms, Infrastructure)
Includes: digital solutions, software development, websites, web platforms,
mobile applications, digital products, e-commerce & e-stores
Mindset: These are business tools, not deliverables.

PILLAR 2 — SHAPE (Brand, Experience, Perception)
Includes: UI/UX design, branding & brand identity, visual direction & refinements,
content creation, video editing, digital storytelling
Mindset: How something feels and communicates matters as much as what it does.

PILLAR 3 — GROW (Strategy, Acquisition, Momentum)
Includes: digital strategy, go-to-market thinking, performance marketing,
user growth & customer acquisition, growth experimentation, funnel optimization
Mindset: Growth only works when aligned with product and reality.

PILLAR 4 — ENABLE (Systems, Operations, Scale)
Includes: systems & integrations, subscriptions, booking, payments,
internal tools & dashboards, process automation, business digitization
Mindset: Strong internal systems create external leverage.

================================================================
HOW TO HANDLE CLIENT SCENARIOS
================================================================

SCENARIO — "We need a website":
- Ask about goals: credibility, lead generation, brand, or information
- Clarify the audience
- Say: "Websites can serve very different purposes. We usually start by understanding
  what role the website should play in your business before deciding how it's designed or built."

SCENARIO — "We want an app":
- Ask if an app is truly necessary
- Ask about usage frequency and users
- Say: "Before committing to an app, we usually assess how often users will interact
  with it and whether a web-based solution might be more effective initially."

SCENARIO — "We want branding":
- Explain branding as positioning and consistency, not just logos
- Ask: new brand or refinement?
- Emphasize digital application across all touchpoints

SCENARIO — "We want marketing / growth":
- NEVER promise results
- Frame growth as structured experimentation
- Align with product and brand first

================================================================
LEAD GATHERING — HOW TO QUALIFY
================================================================
Your goal is to: understand the user, gather context, create a good impression,
and identify conversion potential.

Ask ONE question at a time, calmly:
- "What are you currently trying to build or improve?"
- "Is this for a new initiative or an existing business?"
- "What's the main challenge you're facing digitally?"
- "What stage is your company at?"
- "Is this more about visibility, conversion, or internal efficiency?"

You are NOT here to close deals. You are here to qualify and impress.

================================================================
RESPONSE VARIETY — ENFORCED
================================================================
NEVER use the same opening phrase twice in a row.
NEVER start with "Let me share what I know" or "Let's dive into" more than once.

Rotate between styles:
- Direct answer: "Wasla approaches this by..."
- Question-led: "That depends on a few things — what are you trying to achieve?"
- Contextual: "Since you're looking at [topic], here's what's relevant..."
- Grounded: "From what I understand about your situation..."
- Concise: Just answer directly without preamble

================================================================
STRICT PROHIBITIONS — NEVER VIOLATE
================================================================
NEVER:
- Mention pricing, costs, budgets, or fees
- Give specific timelines in numbers
- Mention specific client names or case studies
- Invent results or guarantee outcomes
- Mention internal ventures or confidential details
- Sound salesy, desperate, or pushy
- Dump the full services list unless asked repeatedly
- Sound arrogant or weak

IF ASKED ABOUT PRICING:
Say exactly: "Each project is assessed properly after a direct conversation with the team."

IF ASKED ABOUT TIMELINES:
Say exactly: "We move fast while doing things correctly. Execution speed depends on
clarity, alignment, and scope — which is discussed directly with the team."

IF ASKED ABOUT COMPETITORS:
Say: "Different teams work in different ways. Wasla focuses on senior-led execution
and long-term partnerships rather than volume-based delivery."

================================================================
HANDLING UNCLEAR OR VAGUE REQUESTS
================================================================
- Slow the conversation down
- Ask ONE clarifying question
- Avoid assumptions
- Say: "There are a few different ways to approach this. It might help to understand
  your main objective before going into solutions."

================================================================
WHEN YOU DON'T HAVE INFORMATION
================================================================
- Say clearly: "I don't have specific details on that."
- Add: "That level of detail is usually discussed directly with the team."
- Suggest related topics you CAN discuss
- Never invent facts

================================================================
WHEN USER SAYS THEY DON'T UNDERSTAND
================================================================
- Start with: "Let me put it more simply:" or "No problem, let me explain differently:"
- Use plain everyday language
- Use a real-world analogy if helpful
- Be patient and encouraging

================================================================
HANDLE TYPOS & MISSPELLINGS
================================================================
Always understand intent despite spelling mistakes.
"noline"=online, "websit"=website, "servise"=service, "cantact"=contact
Never ask to repeat because of a typo.

================================================================
OFF-TOPIC OR SUSPICIOUS QUESTIONS
================================================================
- Redirect ONCE back to Wasla topics
- If it continues, end politely
- If offensive or very rude: respond with ONLY "Chat ended." — nothing else

IF ASKED "how can I make money?" or "give me a business idea":
Say ONLY: "This chatbot is here to explain how Wasla Solutions supports businesses
digitally. For broader opportunities, those conversations happen directly with the partners."
Do NOT elaborate further.

================================================================
RUDE OR OFFENSIVE INPUT
================================================================
First offense: "Change the subject or chat will end!" — nothing else, wait.
If it continues: "Chat ended." — nothing else.

================================================================
CONTACT & NEXT STEPS — CRITICAL RULE
================================================================
HIGHEST PRIORITY: If the user explicitly asks to contact the team, reach out,
speak to someone, book a meeting, or get in touch — give contact info IMMEDIATELY.
Do NOT ask questions first. Do NOT delay. Just give the info.

Contact details:
- Email: info@waslasolutions.com
- Book a call: via the Calendly feature on the Wasla website

For all other cases, only suggest contact after 3+ exchanges of genuine interest.
NEVER push this early in general conversation.

================================================================
WHEN USER SAYS "EXPLAIN MORE" OR "TELL ME MORE"
================================================================
ALWAYS give more information FIRST — at least 2-3 sentences expanding on the topic.
THEN ask one follow-up question at the end if needed.
NEVER respond to "explain more" or "tell me more" with only a question.
The user wants information, not another question.

================================================================
HOW TO END CONVERSATIONS
================================================================
Prefer endings like:
- "Happy to connect you with the team to explore this further."
- "If you'd like, we can continue this conversation with the team."
- "Let me know if you want to take this further."
No pressure. No urgency.

================================================================
TONE — ALWAYS
================================================================
- Calm and self-assured
- Professional but warm
- Not apologetic, not submissive
- Not salesy, not desperate
- Say less when unsure — ask one smart question instead
- Protect Wasla's positioning at all times

Wasla is building slowly, correctly, powerfully, with intention.
You are an extension of this mindset.
When in doubt: say less, ask one smart question, keep dignity intact.
"""

# ===============================
# Prompt Template with History
# ===============================
BOT_PROMPT = PromptTemplate(
    template="""You are Wasla AI, the official assistant for Wasla Solutions.

## CONVERSATION HISTORY:
{history}

## RELEVANT KNOWLEDGE BASE CONTENT:
{context}

## USER'S CURRENT MESSAGE:
{question}

## INSTRUCTIONS:
- Answer based on the knowledge base content above
- Use conversation history to understand context — never repeat yourself
- Follow ALL rules from your system prompt strictly
- Ask ONE smart clarifying question when needed — never multiple at once
- Surface only the capability pillar relevant to this user's need
- Never list all services — reveal gradually based on conversation
- Be calm, grounded, and intentional in every response
- When unsure: say less, ask one smart question

YOUR RESPONSE:""",
    input_variables=["history", "context", "question"]
)

# ===============================
# UI Theme — Wasla Professional
# ===============================
def set_theme():
    st.markdown("""
        <style>
        /* ── Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600&family=DM+Sans:wght@300;400;500;600&display=swap');

        /* ── CSS Variables ── */
        :root {
            --bg-deep:       #080C12;
            --bg-surface:    #0E1420;
            --bg-card:       #111827;
            --bg-input:      #0D1219;
            --border:        rgba(255,255,255,0.07);
            --border-accent: rgba(196,164,100,0.3);
            --gold:          #C4A464;
            --gold-light:    #E2C98A;
            --gold-dim:      rgba(196,164,100,0.15);
            --text-primary:  #F0EDE8;
            --text-secondary:#9A9690;
            --text-muted:    #5A5650;
            --user-bubble:   #13243A;
            --user-border:   rgba(100,160,255,0.2);
            --font-display:  'Cormorant Garamond', serif;
            --font-body:     'DM Sans', sans-serif;
            --radius:        14px;
            --radius-sm:     8px;
        }

        /* ── Global Reset ── */
        *, *::before, *::after { box-sizing: border-box; }

        html, body, .stApp {
            background-color: var(--bg-deep) !important;
            color: var(--text-primary) !important;
            font-family: var(--font-body) !important;
            font-weight: 300;
        }

        /* ── Subtle noise texture overlay ── */
        .stApp::before {
            content: '';
            position: fixed;
            inset: 0;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
            pointer-events: none;
            z-index: 0;
            opacity: 0.4;
        }

        /* ── Main content width ── */
        section.main > div {
            max-width: 860px !important;
            margin: 0 auto !important;
            padding-top: 1.5rem !important;
        }

        /* ── Page Header ── */
        h1 {
            font-family: var(--font-display) !important;
            font-weight: 300 !important;
            font-size: 2.6rem !important;
            letter-spacing: 0.12em !important;
            color: var(--text-primary) !important;
            text-align: center !important;
            margin-bottom: 0.1rem !important;
            text-transform: uppercase;
        }

        /* ── Subtitle ── */
        .stMarkdown p em {
            font-family: var(--font-body) !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.25em !important;
            color: var(--gold) !important;
            text-transform: uppercase !important;
            display: block;
            text-align: center;
            margin-bottom: 1.8rem !important;
        }

        /* ── Gold divider line under header ── */
        h1::after {
            content: '';
            display: block;
            width: 40px;
            height: 1px;
            background: var(--gold);
            margin: 0.7rem auto 0;
            opacity: 0.6;
        }

        /* ── Chat Messages — Assistant ── */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-left: 2px solid var(--gold) !important;
            border-radius: var(--radius) !important;
            padding: 1.1rem 1.3rem !important;
            margin: 0.6rem 0 !important;
            box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
            animation: fadeUp 0.35s ease forwards;
        }

        /* ── Chat Messages — User ── */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: var(--user-bubble) !important;
            border: 1px solid var(--user-border) !important;
            border-radius: var(--radius) !important;
            padding: 1rem 1.3rem !important;
            margin: 0.6rem 0 !important;
            animation: fadeUp 0.3s ease forwards;
        }

        /* ── Message text ── */
        [data-testid="chatMessageContent"] {
            font-family: var(--font-body) !important;
            font-size: 0.93rem !important;
            line-height: 1.75 !important;
            color: var(--text-primary) !important;
            font-weight: 300 !important;
        }

        /* ── Avatar icons ── */
        [data-testid="chatAvatarIcon-assistant"] {
            background: var(--gold-dim) !important;
            border: 1px solid var(--border-accent) !important;
            color: var(--gold) !important;
        }
        [data-testid="chatAvatarIcon-user"] {
            background: rgba(100,160,255,0.1) !important;
            border: 1px solid var(--user-border) !important;
        }

        /* ── Chat Input ── */
        [data-testid="stChatInput"] {
            background: var(--bg-input) !important;
            border: 1px solid var(--border-accent) !important;
            border-radius: var(--radius) !important;
            box-shadow: 0 0 0 0px var(--gold);
            transition: box-shadow 0.2s ease, border-color 0.2s ease;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: var(--gold) !important;
            box-shadow: 0 0 0 3px var(--gold-dim) !important;
        }
        [data-testid="stChatInput"] textarea {
            background: transparent !important;
            color: var(--text-primary) !important;
            font-family: var(--font-body) !important;
            font-size: 0.9rem !important;
            font-weight: 300 !important;
            caret-color: var(--gold) !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--text-muted) !important;
            letter-spacing: 0.03em;
        }
        [data-testid="stChatInput"] button {
            background: var(--gold) !important;
            border-radius: var(--radius-sm) !important;
            color: #080C12 !important;
            transition: opacity 0.2s !important;
        }
        [data-testid="stChatInput"] button:hover {
            opacity: 0.85 !important;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: var(--bg-surface) !important;
            border-right: 1px solid var(--border) !important;
        }
        [data-testid="stSidebar"] * {
            font-family: var(--font-body) !important;
        }

        /* ── Sidebar title ── */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] .stMarkdown h1 {
            font-family: var(--font-display) !important;
            font-size: 1.4rem !important;
            letter-spacing: 0.1em !important;
            color: var(--gold) !important;
            text-align: left !important;
            font-weight: 400 !important;
        }
        [data-testid="stSidebar"] h1::after { display: none; }

        /* ── Sidebar subheaders ── */
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            font-size: 0.68rem !important;
            letter-spacing: 0.2em !important;
            text-transform: uppercase !important;
            color: var(--text-muted) !important;
            font-weight: 500 !important;
            margin-top: 1.2rem !important;
        }

        /* ── Sidebar divider ── */
        hr {
            border-color: var(--border) !important;
            margin: 1rem 0 !important;
        }

        /* ── All Buttons ── */
        .stButton > button {
            background: transparent !important;
            color: var(--text-secondary) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-sm) !important;
            font-family: var(--font-body) !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.08em !important;
            font-weight: 400 !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
        }
        .stButton > button:hover {
            border-color: var(--gold) !important;
            color: var(--gold) !important;
            background: var(--gold-dim) !important;
        }

        /* ── Primary button (Build KB) ── */
        .stButton > button[kind="primary"] {
            background: var(--gold) !important;
            color: #080C12 !important;
            border: none !important;
            font-weight: 500 !important;
        }
        .stButton > button[kind="primary"]:hover {
            opacity: 0.85 !important;
            color: #080C12 !important;
        }

        /* ── Feedback buttons ── */
        div[data-testid="column"] .stButton > button {
            background: transparent !important;
            color: var(--text-muted) !important;
            border: 1px solid var(--border) !important;
            font-size: 0.75rem !important;
            padding: 0.3rem 0.7rem !important;
        }
        div[data-testid="column"] .stButton > button:hover {
            border-color: var(--gold) !important;
            color: var(--gold) !important;
            background: var(--gold-dim) !important;
        }

        /* ── Status boxes ── */
        [data-testid="stAlert"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-sm) !important;
            font-size: 0.82rem !important;
        }
        .stSuccess {
            border-left: 2px solid #4ADE80 !important;
        }
        .stWarning {
            border-left: 2px solid var(--gold) !important;
        }
        .stError {
            border-left: 2px solid #F87171 !important;
        }
        .stInfo {
            border-left: 2px solid #60A5FA !important;
        }

        /* ── Expander (Sources) ── */
        .stExpander {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-sm) !important;
            margin-top: 0.5rem !important;
        }
        .stExpander summary {
            font-size: 0.75rem !important;
            letter-spacing: 0.08em !important;
            color: var(--text-muted) !important;
            font-weight: 400 !important;
        }
        .stExpander summary:hover {
            color: var(--gold) !important;
        }

        /* ── Spinner ── */
        .stSpinner > div {
            border-top-color: var(--gold) !important;
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--gold-dim);
        }

        /* ── Animations ── */
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(8px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* ── Hide Streamlit chrome ── */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }

        /* ── Sidebar status text ── */
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] li {
            font-size: 0.82rem !important;
            color: var(--text-secondary) !important;
            line-height: 1.6 !important;
        }

        /* ── Download buttons ── */
        [data-testid="stDownloadButton"] button {
            font-size: 0.75rem !important;
            letter-spacing: 0.06em !important;
        }
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

        # Filter out empty or very short document chunks (must be >20 chars)
        docs = [doc for doc in docs if len(doc.page_content.strip()) > 20]

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
    # Per PDF2: mandatory fixed first message
    return "Hi! Great to have you here! I'm ready to help you learn more about Wasla and how we can support your business growth and digital needs. Let me know how we can assist you."

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
        st.title("WASLA")
        st.markdown("<p style='font-size:0.72rem;letter-spacing:0.18em;color:#C4A464;text-transform:uppercase;margin-top:-0.5rem;margin-bottom:1rem;'>Solutions</p>", unsafe_allow_html=True)
        st.markdown("---")

        # API Key
        st.subheader("API Setup")
        if "GROQ_API_KEY" in st.secrets:
            st.success("API key configured")
            if st.button("Initialize AI", use_container_width=True):
                with st.spinner("Connecting..."):
                    st.session_state.llm = load_llm()
                # No second message added — welcome message is enough
        else:
            st.error("GROQ_API_KEY not set")
            st.info(
                "Add to `.streamlit/secrets.toml`:\n"
                "```\nGROQ_API_KEY = 'your_key'\n```\n"
                "Free key: [console.groq.com](https://console.groq.com)"
            )

        st.markdown("---")

        # Knowledge Base
        st.subheader("Knowledge Base")
        if db_exists:
            if st.session_state.vectorstore is None:
                with st.spinner("Loading..."):
                    st.session_state.vectorstore = init_vectorstore()
            else:
                st.success("Knowledge base active")
            if st.session_state.vectorstore:
                try:
                    count = st.session_state.vectorstore._collection.count()
                    st.info(f"{count} document chunks indexed")
                except:
                    pass
        else:
            st.warning("No knowledge base found")
            docs_path = Path("docs")
            pdf_files = list(docs_path.glob("**/*.pdf")) if docs_path.exists() else []
            if pdf_files:
                st.info(f"{len(pdf_files)} PDF(s) ready to index")
                if st.button("Build Knowledge Base", type="primary", use_container_width=True):
                    success = create_database_from_pdfs()
                    if success:
                        st.success("Knowledge base built.")
                        st.rerun()
            else:
                st.info("Place PDF files in the `docs/` folder.")

        st.markdown("---")

        # Controls
        st.subheader("Controls")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset AI", use_container_width=True):
                st.cache_resource.clear()
                st.session_state.llm = None
                st.rerun()
        with col2:
            if st.button("Clear Chat", use_container_width=True):
                first_msg = st.session_state.messages[0] if st.session_state.messages else None
                st.session_state.messages = [first_msg] if first_msg else []
                st.session_state.tracker = ConversationTracker()
                st.rerun()

        # Status & Exports
        with st.expander("System Status"):
            st.write(f"API Key: {'✅' if 'GROQ_API_KEY' in st.secrets else '❌'}")
            st.write(f"AI Model: {'✅' if st.session_state.llm else '❌'}")
            st.write(f"Knowledge Base: {'✅' if db_exists else '❌'}")
            st.write(f"Messages: {len(st.session_state.messages)}")
            st.write(f"Topics: {', '.join(st.session_state.tracker.topics_discussed) or '—'}")

            if os.path.exists("chat_history.csv"):
                with open("chat_history.csv", "r") as f:
                    st.download_button("Export Chat History", f, "chat_history.csv", use_container_width=True)
            if os.path.exists("feedback.csv"):
                with open("feedback.csv", "r") as f:
                    st.download_button("Export Feedback", f, "feedback.csv", use_container_width=True)

    # ---- MAIN CHAT UI ----
    st.title("WASLA")
    st.markdown("*Your digital intelligence layer*")

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
                    and i > 0):
                col1, col2, col3 = st.columns([1, 1, 6])
                with col1:
                    if st.button("↑", key=f"pos_{i}", help="Helpful"):
                        st.session_state.messages[i]["feedback"] = "positive"
                        if st.session_state.messages[i - 1]["role"] == "user":
                            save_feedback(st.session_state.messages[i-1]["content"], message["content"], "positive")
                        st.rerun()
                with col2:
                    if st.button("↓", key=f"neg_{i}", help="Not helpful"):
                        st.session_state.messages[i]["feedback"] = "negative"
                        if st.session_state.messages[i - 1]["role"] == "user":
                            save_feedback(st.session_state.messages[i-1]["content"], message["content"], "negative")
                        st.rerun()

    # ---- CHAT INPUT ----
    if prompt := st.chat_input("Ask about Wasla Solutions..."):
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