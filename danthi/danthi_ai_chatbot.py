"""
          DANTHI - AI CHATBOT - ALL-IN-ONE               
     Features: Chat | Voice | Web Scraper | Document Q&A         
"""

import base64
import os
import threading
from pathlib import Path

import faiss
import numpy as np
import PyPDF2
import requests
import pyttsx3
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import CharacterTextSplitter

# ──────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="🌸 Danthi AI",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
#  LOAD OWL IMAGE AS BASE64
# ──────────────────────────────────────────────
def get_owl_base64() -> str:
    assets_dir = Path("assets")
    candidates = [
        assets_dir / "owl.png",
        assets_dir / "owl.gif",
        assets_dir / "owl.jpg",
        assets_dir / "owl.jpeg",
        assets_dir / "owl.webp",
    ]
    if assets_dir.exists():
        for f in assets_dir.iterdir():
            if f.suffix.lower() in (".png", ".gif", ".jpg", ".jpeg", ".webp"):
                candidates.append(f)

    for path in candidates:
        if path.exists():
            ext = path.suffix.lower()
            mime_map = {".png": "image/png", ".gif": "image/gif",
                        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
            mime = mime_map.get(ext, "image/png")
            data = base64.b64encode(path.read_bytes()).decode()
            return f"data:{mime};base64,{data}"
    return ""

OWL_SRC = get_owl_base64()

# ──────────────────────────────────────────────
#  GLOBAL CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

.chat-bubble-user {
    background: linear-gradient(135deg, #185fa5, #2e86de);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 6px 0 6px auto;
    max-width: 78%;
    display: block;
    word-wrap: break-word;
    line-height: 1.55;
}
.chat-bubble-ai {
    background: #f0f4f8;
    color: #1a1a2e;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px auto 6px 0;
    max-width: 78%;
    display: block;
    word-wrap: break-word;
    line-height: 1.55;
}
.feature-card {
    background: linear-gradient(135deg, #f8fbff 0%, #e8f0fb 100%);
    border-radius: 14px;
    padding: 18px 22px;
    margin: 10px 0;
    border-left: 5px solid #185fa5;
    color: #1a2a3a;
    line-height: 1.75;
    font-size: 0.97rem;
}
.stButton > button {
    background: linear-gradient(135deg, #185fa5 0%, #2e86de 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.5rem 1.6rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(24,95,165,0.28) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 20px rgba(24,95,165,0.38) !important;
}
.stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 1.5px solid #c0d8f5 !important;
    font-family: 'Poppins', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: #185fa5 !important;
    box-shadow: 0 0 0 2px rgba(24,95,165,0.15) !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""
if "va_status" not in st.session_state:
    st.session_state.va_status = "idle"
if "va_pending_q" not in st.session_state:
    st.session_state.va_pending_q = ""
if "va_typed_q" not in st.session_state:
    st.session_state.va_typed_q = ""
if "va_last_response" not in st.session_state:
    st.session_state.va_last_response = ""
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = faiss.IndexFlatL2(384)
if "vector_store" not in st.session_state:
    st.session_state.vector_store = {}
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False

# ──────────────────────────────────────────────
#  LOAD MODELS
# ──────────────────────────────────────────────

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def get_llm():
    """
    Always return a FRESH OllamaLLM instance to avoid stale httpx client errors.
    """
    return OllamaLLM(
        model="mistral",
        base_url="http://localhost:11434",
        timeout=120,
    )

def check_ollama_alive() -> bool:
    try:
        r = requests.get("http://localhost:11434", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

try:
    embeddings = load_embeddings()
    EMBED_OK = True
except Exception as e:
    embeddings = None
    EMBED_OK = False
    st.warning(f"⚠️ Embeddings failed to load: {e}")

OLLAMA_OK = check_ollama_alive()
MODEL_OK  = EMBED_OK

if not OLLAMA_OK:
    st.error(
        "❌ **Ollama server is not running!**\n\n"
        "Terminal එකේ මේ commands run කරන්න:\n"
        "```\nollama pull mistral\nollama serve\n```\n\n"
        "Ollama start කළාට පස්සේ page එක refresh කරන්න. 🔄"
    )
    st.info("💡 Ollama install නැත්නම්: https://ollama.com/download")
    MODEL_OK = False

# ──────────────────────────────────────────────
#  PROMPT TEMPLATE
# ──────────────────────────────────────────────
prompt_template = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=(
        "You are Danthi, a friendly, smart, and helpful AI assistant.\n"
        "Previous conversation:\n{chat_history}\n\n"
        "User: {question}\nDanthi:"
    )
)

# ──────────────────────────────────────────────
#  CORE FUNCTIONS
# ──────────────────────────────────────────────

def run_chat(question: str) -> str:
    """💬 Chat with memory — retries once on closed-client error"""
    history_text = "\n".join(
        f"{m.type.capitalize()}: {m.content}"
        for m in st.session_state.chat_history.messages
    )
    prompt_text = prompt_template.format(
        chat_history=history_text, question=question
    )
    for attempt in range(2):
        try:
            response = get_llm().invoke(prompt_text)
            st.session_state.chat_history.add_user_message(question)
            st.session_state.chat_history.add_ai_message(response)
            return response
        except Exception as e:
            if attempt == 0:
                # ✅ BUG FIX 1: st.cache_resource.clear() වෙනුවට load_embeddings.clear()
                load_embeddings.clear()
                continue
            return (
                f"⚠️ Danthi could not connect to Ollama.\n\n"
                f"Error: {e}\n\n"
                "Please make sure `ollama serve` is running, then refresh the page."
            )


def speak_text(text: str):
    """🔊 TTS — synchronous, reliable on Windows"""
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 155)
        engine.setProperty("volume", 1.0)
        # Pick a clear voice if available
        voices = engine.getProperty("voices")
        if voices:
            engine.setProperty("voice", voices[0].id)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        st.warning(f"🔊 TTS error: {e}")


def listen_microphone() -> str:
    """
    🎙️ Real microphone input using SpeechRecognition library.
    Works 100% on Windows — no browser restrictions.
    Returns recognised text or error string starting with 'ERROR:'.
    """
    try:
        import speech_recognition as sr
    except ImportError:
        return "ERROR: SpeechRecognition library not installed. Run: py -3.10 -m pip install SpeechRecognition pyaudio"

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=15)
        text = recognizer.recognize_google(audio, language="en-US")
        return text
    except sr.WaitTimeoutError:
        return "ERROR: No speech detected. Please speak louder or check mic."
    except sr.UnknownValueError:
        return "ERROR: Could not understand. Please speak clearly."
    except sr.RequestError as e:
        return f"ERROR: Google Speech API error: {e}"
    except OSError:
        return "ERROR: Microphone not found. Check mic connection."
    except Exception as e:
        return f"ERROR: {e}"


def scrape_website(url: str):
    """🌐 Scrape paragraph text — returns (success: bool, content: str)"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return False, f"Failed to fetch page (HTTP {resp.status_code})"
        soup = BeautifulSoup(resp.text, "html.parser")
        text = " ".join(p.get_text() for p in soup.find_all("p"))
        if not text.strip():
            return False, "No readable text found on this page."
        return True, text[:3000]
    except Exception as e:
        return False, str(e)


def summarize_web_content(content: str) -> str:
    return get_llm().invoke(
        f"Summarize the following web page content clearly and concisely:\n\n{content[:2000]}"
    )


def extract_pdf_text(uploaded_file) -> str:
    """📄 Extract all text from PDF"""
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def store_doc_in_faiss(text: str, filename: str) -> str:
    """📦 Chunk + embed + store in session-scoped FAISS"""
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    if not chunks:
        return "⚠️ No text found in the document."
    vectors = np.array(embeddings.embed_documents(chunks), dtype=np.float32)
    idx_start = st.session_state.faiss_index.ntotal
    st.session_state.faiss_index.add(vectors)
    key = len(st.session_state.vector_store)
    st.session_state.vector_store[key] = (filename, chunks, idx_start)
    st.session_state.doc_loaded = True
    return f"✅ '{filename}' indexed! ({len(chunks)} chunks)"


def answer_from_doc(query: str) -> str:
    """🔍 Semantic search → LLM answer from document"""
    if not st.session_state.doc_loaded:
        return "⚠️ No document loaded. Please upload a PDF first."
    q_vec = np.array(
        embeddings.embed_query(query), dtype=np.float32
    ).reshape(1, -1)
    _, indices = st.session_state.faiss_index.search(q_vec, k=3)
    context = ""
    for idx in indices[0]:
        # ✅ BUG FIX 2: FAISS -1 result (not found) crash fix
        if idx == -1:
            continue
        for fname, chunks, start in st.session_state.vector_store.values():
            local_idx = idx - start
            if 0 <= local_idx < len(chunks):
                context += chunks[local_idx] + "\n\n"
    if not context.strip():
        return "🧠 No relevant content found in the document."
    return get_llm().invoke(
        f"Answer the question using ONLY the document context below.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )


# ──────────────────────────────────────────────
#  3D OWL HERO HTML
# ──────────────────────────────────────────────
def owl_hero_html(owl_src: str) -> str:
    owl_content = (
        f'<img id="owlImg" src="{owl_src}" alt="Danthi owl mascot" draggable="false" />'
        if owl_src else
        '<div id="owlImg" style="font-size:120px;line-height:1;user-select:none;">🦉</div>'
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: transparent;
    font-family: 'Poppins', 'Segoe UI', sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 390px;
    overflow: hidden;
    position: relative;
  }}

  .rings {{
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: none;
  }}
  .ring {{
    position: absolute;
    border-radius: 50%;
    border: 1px solid rgba(24,95,165,0.10);
  }}
  .ring:nth-child(1) {{ width: 240px; height: 240px; }}
  .ring:nth-child(2) {{ width: 350px; height: 350px; border-color: rgba(24,95,165,0.06); }}
  .ring:nth-child(3) {{ width: 460px; height: 460px; border-color: rgba(24,95,165,0.035); }}

  .owl-wrapper {{
    perspective: 800px;
    cursor: grab;
    user-select: none;
    -webkit-user-select: none;
    z-index: 2;
    position: relative;
  }}
  .owl-wrapper:active {{ cursor: grabbing; }}

  .owl-inner {{
    width: 200px;
    height: 200px;
    transform-style: preserve-3d;
    animation: floatBob 4s ease-in-out infinite;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
  }}

  #owlImg {{
    width: 200px;
    height: 200px;
    object-fit: contain;
    display: block;
    pointer-events: none;
    filter: drop-shadow(0 18px 32px rgba(24,95,165,0.26)) drop-shadow(0 4px 8px rgba(24,95,165,0.14));
  }}

  .glow {{
    position: absolute;
    bottom: -10px;
    left: 50%;
    transform: translateX(-50%);
    width: 130px; height: 26px;
    background: radial-gradient(ellipse, rgba(24,95,165,0.25) 0%, transparent 75%);
    border-radius: 50%;
    filter: blur(5px);
    animation: glowP 4s ease-in-out infinite;
    pointer-events: none;
  }}
  @keyframes glowP {{
    0%,100% {{ opacity:.7; transform:translateX(-50%) scaleX(1); }}
    50%      {{ opacity:.28; transform:translateX(-50%) scaleX(.68); }}
  }}

  .ground-shadow {{
    width: 140px; height: 14px;
    background: radial-gradient(ellipse, rgba(24,95,165,0.13) 0%, transparent 72%);
    border-radius: 50%;
    margin-top: 2px;
    animation: shadowA 4s ease-in-out infinite;
    z-index: 2;
  }}
  @keyframes shadowA {{
    0%,100% {{ transform:scaleX(1); opacity:1; }}
    50%      {{ transform:scaleX(.65); opacity:.4; }}
  }}

  @keyframes floatBob {{
    0%,100% {{ transform: translateY(0)    rotateY(0deg)  rotateX(2deg); }}
    25%      {{ transform: translateY(-11px) rotateY(6deg)  rotateX(1deg); }}
    50%      {{ transform: translateY(-17px) rotateY(0deg)  rotateX(0deg); }}
    75%      {{ transform: translateY(-11px) rotateY(-6deg) rotateX(1deg); }}
  }}

  .bubble {{
    position: absolute;
    top: -52px; left: 50%;
    transform: translateX(-50%) scale(0);
    background: #185fa5;
    color: #fff;
    font-size: 13px;
    font-weight: 500;
    padding: 8px 16px;
    border-radius: 20px;
    white-space: nowrap;
    opacity: 0;
    transition: transform .3s cubic-bezier(.34,1.56,.64,1), opacity .25s;
    pointer-events: none;
    z-index: 10;
  }}
  .bubble::after {{
    content: '';
    position: absolute;
    bottom: -7px; left: 50%;
    transform: translateX(-50%);
    border: 7px solid transparent;
    border-top-color: #185fa5;
    border-bottom: none;
  }}
  .bubble.show {{
    transform: translateX(-50%) scale(1);
    opacity: 1;
  }}

  .spinning .owl-inner {{
    animation: spin360 .95s cubic-bezier(.4,0,.2,1) forwards !important;
  }}
  @keyframes spin360 {{
    0%   {{ transform: rotateY(0deg)   scale(1); }}
    50%  {{ transform: rotateY(180deg) scale(1.1); }}
    100% {{ transform: rotateY(360deg) scale(1); }}
  }}

  .name-block {{
    text-align: center;
    margin-top: 1rem;
    z-index: 2;
  }}
  .name-title {{
    font-size: 2rem;
    font-weight: 700;
    color: #1a1a2e;
    letter-spacing: -0.5px;
  }}
  .name-title .accent {{ color: #185fa5; }}
  .name-sub {{
    font-size: 0.82rem;
    color: #6b7a8d;
    margin-top: 3px;
    letter-spacing: 0.07em;
    text-transform: uppercase;
  }}

  .chips {{
    display: flex;
    gap: 7px;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 0.9rem;
    z-index: 2;
  }}
  .chip {{
    display: flex;
    align-items: center;
    gap: 5px;
    background: #f0f4f8;
    border: 1px solid #d0e3f5;
    border-radius: 999px;
    padding: 5px 13px;
    font-size: 12px;
    color: #3a5a7a;
    cursor: default;
    transition: background .15s;
  }}
  .chip:hover {{ background: #daeaf7; }}

  .action-row {{
    display: flex;
    gap: 9px;
    margin-top: 1rem;
    z-index: 2;
  }}
  .btn {{
    display: flex;
    align-items: center;
    gap: 6px;
    border-radius: 8px;
    padding: 7px 18px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    border: none;
    transition: opacity .15s, transform .15s;
  }}
  .btn:hover {{ opacity: .88; transform: translateY(-1px); }}
  .btn-outline {{
    background: #fff;
    border: 1px solid #c0d8f5;
    color: #185fa5;
  }}
  .btn-solid {{
    background: linear-gradient(135deg, #185fa5, #2e86de);
    color: #fff;
  }}
</style>
</head>
<body>

<div class="rings">
  <div class="ring"></div>
  <div class="ring"></div>
  <div class="ring"></div>
</div>

<div class="owl-wrapper" id="owlWrapper">
  <div class="owl-inner" id="owlInner">
    {owl_content}
    <div class="glow"></div>
    <div class="bubble" id="bubble">Hi! I'm Danthi! 🌸</div>
  </div>
</div>
<div class="ground-shadow"></div>

<div class="name-block">
  <div class="name-title"><span class="accent">Danthi</span> AI</div>
  <div class="name-sub">Your Smart Assistant</div>
</div>

<div class="chips">
  <div class="chip">💬 Chat</div>
  <div class="chip">🎙️ Voice</div>
  <div class="chip">🌐 Web Scraper</div>
  <div class="chip">📄 Document Q&A</div>
</div>

<div class="action-row">
  <button class="btn btn-outline" onclick="spinOwl()">🔄 Spin</button>
  <button class="btn btn-solid"   onclick="waveOwl()">👋 Say Hi</button>
</div>

<script>
  const wrapper = document.getElementById('owlWrapper');
  const inner   = document.getElementById('owlInner');
  const bubble  = document.getElementById('bubble');
  let dragging=false, startX=0, rotY=0;

  wrapper.addEventListener('mousedown', e=>{{
    dragging=true; startX=e.clientX;
    inner.style.animation='none';
  }});
  window.addEventListener('mousemove', e=>{{
    if(!dragging) return;
    rotY+=(e.clientX-startX)*0.65;
    inner.style.transform='rotateY('+rotY+'deg)';
    startX=e.clientX;
  }});
  window.addEventListener('mouseup', ()=>{{
    if(!dragging) return; dragging=false;
    setTimeout(()=>{{ inner.style.transform=''; inner.style.animation=''; rotY=0; }}, 1300);
  }});

  wrapper.addEventListener('touchstart', e=>{{
    startX=e.touches[0].clientX; inner.style.animation='none';
  }},{{passive:true}});
  wrapper.addEventListener('touchmove', e=>{{
    rotY+=(e.touches[0].clientX-startX)*0.5;
    inner.style.transform='rotateY('+rotY+'deg)';
    startX=e.touches[0].clientX;
  }},{{passive:true}});
  wrapper.addEventListener('touchend', ()=>{{
    setTimeout(()=>{{ inner.style.transform=''; inner.style.animation=''; rotY=0; }},1300);
  }});

  function spinOwl(){{
    inner.style.animation='none';
    wrapper.classList.add('spinning');
    setTimeout(()=>{{ wrapper.classList.remove('spinning'); inner.style.animation=''; rotY=0; }},970);
  }}

  function waveOwl(){{
    bubble.classList.add('show');
    inner.style.animation='none';
    const frames=[
      [0,   'rotateZ(-12deg) scale(1.07)'],
      [200, 'rotateZ(12deg)  scale(1.07)'],
      [400, 'rotateZ(-7deg)  scale(1.03)'],
      [600, 'rotateZ(0deg)   scale(1)']
    ];
    frames.forEach(([t,v])=>setTimeout(()=>{{inner.style.transform=v;}},t));
    setTimeout(()=>{{inner.style.animation=''; inner.style.transform=''; rotY=0;}},650);
    setTimeout(()=>bubble.classList.remove('show'), 2800);
  }}
</script>
</body>
</html>"""


# ──────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌸 Danthi AI")
    st.markdown("*Your personal AI assistant*")
    st.markdown("---")

    mode = st.radio(
        "🎛️ Select Mode / Mode තෝරන්න",
        options=["💬 Chat", "🎙️ Voice Assistant", "🌐 Web Scraper", "📄 Document Q&A"],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    tts_enabled = st.toggle("🔊 Voice Reply (TTS)", value=False, key="tts_toggle")

    st.markdown("---")
    st.markdown("### 📌 Mode Guide")
    guides = {
        "💬 Chat":            "Type any question — Danthi remembers everything.\nඕනෑම ප්‍රශ්නයක් type කරන්න.",
        "🎙️ Voice Assistant": "Click mic → speak → Danthi replies.\nButton click කර කතා කරන්න.",
        "🌐 Web Scraper":     "Paste a URL → Danthi summarizes it.\nURL දෙන්න → AI summarize කරයි.",
        "📄 Document Q&A":    "Upload PDF → ask questions about it.\nPDF upload කර ප්‍රශ්න අහන්න."
    }
    st.info(guides.get(mode, ""))

    st.markdown("---")
    if st.button("🗑️ Clear History / History මකන්න"):
        st.session_state.chat_history = ChatMessageHistory()
        st.rerun()

# ──────────────────────────────────────────────
#  3D OWL HERO
# ──────────────────────────────────────────────
components.html(owl_hero_html(OWL_SRC), height=400, scrolling=False)
st.markdown("---")

if not MODEL_OK:
    st.stop()

# ══════════════════════════════════════════════
#  MODE 1 — CHAT
# ══════════════════════════════════════════════
if mode == "💬 Chat":
    st.subheader("💬 Chat with Danthi")
    st.caption("Danthi ට ඔබේ සියලු conversations මතකයි.")

    for msg in st.session_state.chat_history.messages:
        css  = "chat-bubble-user" if msg.type == "human" else "chat-bubble-ai"
        icon = "🧑" if msg.type == "human" else "🌸"
        st.markdown(f'<div class="{css}">{icon} {msg.content}</div>', unsafe_allow_html=True)

    user_input = st.text_input("📝 Ask Danthi anything / ඔබේ ප්‍රශ්නය:", key="chat_input")
    if st.button("Send / යවන්න", key="send_chat") and user_input.strip():
        with st.spinner("🌸 Danthi is thinking..."):
            response = run_chat(user_input.strip())
        if tts_enabled:
            speak_text(response)
        st.rerun()

# ══════════════════════════════════════════════
#  MODE 2 — VOICE ASSISTANT (Continuous Loop)
# ══════════════════════════════════════════════
elif mode == "🎙️ Voice Assistant":
    st.subheader("🎙️ Danthi Voice Assistant")

    # ── Status banner ──────────────────────────
    va_status = st.session_state.get("va_status", "idle")
    status_cfg = {
        "idle":      ("#e8f0fb", "#185fa5", "⚪ Ready — Press START to begin continuous listening"),
        "listening": ("#fff3e0", "#e65100", "🔴 Listening... කතා කරන්න!"),
        "thinking":  ("#f3e5f5", "#6a1b9a", "🧠 Danthi is thinking..."),
        "speaking":  ("#e8f5e9", "#1b5e20", "🔊 Danthi is speaking..."),
    }
    bg, fg, label = status_cfg.get(va_status, status_cfg["idle"])
    st.markdown(
        f"<div style='background:{bg};color:{fg};border-radius:10px;padding:14px 20px;"
        f"font-weight:700;font-size:16px;text-align:center;margin-bottom:14px;"
        f"border:2px solid {fg}33;'>{label}</div>",
        unsafe_allow_html=True
    )

    col_left, col_right = st.columns([1, 2])

    with col_left:
        # Auto-loop toggle
        auto_loop = st.toggle("🔄 Auto-loop (continuous)", value=True, key="va_auto_loop")

        st.markdown("")

        # START button — only show when idle
        if va_status == "idle":
            if st.button("🎙️ START Listening", key="va_start", use_container_width=True):
                st.session_state["va_status"] = "listening"
                st.rerun()
        else:
            if st.button("🛑 STOP", key="va_stop", use_container_width=True):
                st.session_state["va_status"] = "idle"
                st.rerun()

        st.markdown("---")
        st.markdown("**⌨️ Type instead:**")
        type_input = st.text_input("", key="va_type_input", placeholder="Type message here...")
        if st.button("Send ➤", key="va_send_typed", use_container_width=True) and type_input.strip():
            st.session_state["va_typed_q"] = type_input.strip()
            st.session_state["va_status"] = "thinking"
            st.rerun()

        st.markdown("---")
        st.markdown(
            "<div style='font-size:12px;color:#888;line-height:1.9;background:#f8f9fa;"
            "border-radius:8px;padding:10px;'>"
            "1️⃣ <b>START</b> click කරන්න<br>"
            "2️⃣ Clearly කතා කරන්න<br>"
            "3️⃣ Danthi reply දෙයි 🌸<br>"
            "4️⃣ Auto-loop ON නම් නැවත listens"
            "</div>",
            unsafe_allow_html=True
        )

    with col_right:
        st.markdown("### 💬 Conversation")
        msgs = st.session_state.chat_history.messages
        if msgs:
            for msg in msgs:
                icon = "🧑" if msg.type == "human" else "🌸"
                css  = "chat-bubble-user" if msg.type == "human" else "chat-bubble-ai"
                st.markdown(f'<div class="{css}">{icon} {msg.content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='color:#aaa;text-align:center;margin-top:40px;font-size:14px;'>"
                "🎙️ Press START and speak to Danthi!</div>",
                unsafe_allow_html=True
            )

    # ── STATE MACHINE ──────────────────────────────────
    if va_status == "listening":
        with st.spinner("🔴 Listening... Speak clearly!"):
            result = listen_microphone()

        if result.startswith("ERROR:"):
            st.error(f"❌ {result[6:]}")
            st.session_state["va_status"] = "idle"
            st.rerun()
        else:
            st.session_state["va_pending_q"] = result
            st.session_state["va_status"] = "thinking"
            st.rerun()

    elif va_status == "thinking":
        question = st.session_state.pop("va_pending_q",
                   st.session_state.pop("va_typed_q", ""))
        if question:
            with st.spinner(f"🧠 Danthi thinking: *{question}*"):
                ai_response = run_chat(question)
            st.session_state["va_last_response"] = ai_response
            st.session_state["va_status"] = "speaking"
            st.rerun()
        else:
            st.session_state["va_status"] = "idle"
            st.rerun()

    elif va_status == "speaking":
        ai_response = st.session_state.get("va_last_response", "")
        if ai_response:
            st.info(f"🌸 Danthi: {ai_response}")
            with st.spinner("🔊 Danthi is speaking..."):
                speak_text(ai_response)

        # After speaking — auto loop back to listening OR go idle
        if st.session_state.get("va_auto_loop", True):
            st.session_state["va_status"] = "listening"
        else:
            st.session_state["va_status"] = "idle"
        st.rerun()

# ══════════════════════════════════════════════
#  MODE 3 — WEB SCRAPER
# ══════════════════════════════════════════════
elif mode == "🌐 Web Scraper":
    st.subheader("🌐 Danthi Web Scraper")
    st.caption("ඕනෑම website URL එකක් දෙන්න. Danthi ඒක summarize කරලා දෙයි.")

    url      = st.text_input("🔗 Website URL:", key="scraper_url")
    followup = st.text_input(
        "💬 Specific question about this page? (optional) / Specific ප්‍රශ්නයක් (optional):",
        key="scraper_question"
    )

    if st.button("🕷️ Scrape & Summarize", key="scrape_btn") and url.strip():
        with st.spinner("🕷️ Scraping..."):
            success, content = scrape_website(url.strip())

        if not success:
            st.error(f"❌ {content}")
        else:
            st.success(f"✅ Read {len(content)} characters from the page.")
            with st.spinner("🌸 Danthi is summarizing..."):
                if followup.strip():
                    summary = get_llm().invoke(
                        f"Based on this web content, answer: {followup}\n\nContent:\n{content[:2000]}"
                    )
                else:
                    summary = summarize_web_content(content)

            st.markdown("### 📝 Danthi's Summary")
            st.markdown(f'<div class="feature-card">{summary}</div>', unsafe_allow_html=True)
            if tts_enabled:
                speak_text(summary)

            q = followup.strip() if followup.strip() else f"Summarize {url}"
            st.session_state.chat_history.add_user_message(q)
            st.session_state.chat_history.add_ai_message(summary)

# ══════════════════════════════════════════════
#  MODE 4 — DOCUMENT Q&A
# ══════════════════════════════════════════════
elif mode == "📄 Document Q&A":
    st.subheader("📄 Danthi Document Q&A")
    st.caption("PDF upload කරලා Danthi ගෙන් ඒ document ගැන ප්‍රශ්න අහන්න.")

    uploaded_file = st.file_uploader("📁 Upload PDF:", type=["pdf"])

    if uploaded_file:
        file_key = f"pdf_{uploaded_file.name}_{uploaded_file.size}"
        if file_key not in st.session_state:
            with st.spinner(f"📄 Reading '{uploaded_file.name}'..."):
                text = extract_pdf_text(uploaded_file)
            if text.strip():
                result = store_doc_in_faiss(text, uploaded_file.name)
                st.success(result)
                st.session_state[file_key] = True
            else:
                st.error("⚠️ Could not extract text from this PDF.")
        else:
            st.info(f"✅ '{uploaded_file.name}' is already loaded.")

    if st.session_state.doc_loaded:
        loaded_names = list({v[0] for v in st.session_state.vector_store.values()})
        st.caption(f"📚 Loaded: {', '.join(loaded_names)}")

    st.markdown("---")
    query = st.text_input("❓ Ask Danthi about the document:", key="doc_query")

    if st.button("🔍 Get Answer", key="doc_btn") and query.strip():
        with st.spinner("🔍 Danthi is searching the document..."):
            answer = answer_from_doc(query.strip())
        st.markdown("### 🌸 Danthi's Answer:")
        st.markdown(f'<div class="feature-card">{answer}</div>', unsafe_allow_html=True)
        if tts_enabled:
            speak_text(answer)
        st.session_state.chat_history.add_user_message(query.strip())
        st.session_state.chat_history.add_ai_message(answer)

# ──────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#b0bec5; font-size:0.82rem;'>"
    "🌸 Danthi AI · Powered by Ollama (Mistral) + LangChain + FAISS + HuggingFace"
    "</p>",
    unsafe_allow_html=True
)
