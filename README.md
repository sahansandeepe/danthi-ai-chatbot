# 🌸 Danthi AI — Your Personal Voice Assistant

Danthi is an AI-powered voice chatbot that runs completely on your own computer.
No internet subscription. No API fees. Just open it and talk.

Built by **Sahan Sandeepa Ellawala** after completing the *Mastering AI Agents Bootcamp* on Udemy.

---

## ✨ What Danthi Can Do

**💬 Smart Chat**
Talk to Danthi about anything. She remembers everything you said earlier in the conversation.

**🎙️ Voice Assistant**
Press one button, speak your question, and Danthi answers you out loud. Fully hands-free loop — she listens again automatically after each reply.

**🌐 Web Scraper**
Paste any website URL. Danthi reads the page and gives you a clean summary in seconds.

**📄 Document Q&A**
Upload any PDF file. Ask Danthi questions about it, and she finds the answers for you.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python + Streamlit | App interface |
| Ollama + Mistral | Local AI brain (no API cost) |
| LangChain | Memory and prompt management |
| FAISS + HuggingFace | Document search |
| SpeechRecognition + pyttsx3 | Voice input and output |

---

## How to Run This Project

### Step 1 — Install Python
Download Python 3.10 from https://www.python.org/downloads/

> During install, tick **"Add Python to PATH"**

### Step 2 — Install Required Packages
Open PowerShell or Terminal and run:

```bash
pip install streamlit langchain langchain-community langchain-ollama
pip install langchain-huggingface langchain-text-splitters
pip install pyttsx3 requests beautifulsoup4
pip install faiss-cpu numpy PyPDF2
pip install sentence-transformers
pip install SpeechRecognition pyaudio
```

### Step 3 — Install Ollama
Download from https://ollama.com/download and install it.

Then open a terminal and run:
```bash
ollama pull mistral
```
This downloads the AI model (~4GB). Wait for it to finish.

### Step 4 — Start Ollama
```bash
ollama serve
```
> Keep this terminal window open. Do not close it.

### Step 5 — Run the App
Open a new terminal window and run:
```bash
cd path/to/this/folder
python -m streamlit run danthi_ai_chatbot.py
```

Your browser will open at `http://localhost:8501` 🎉

---

## 📁 Folder Structure

```
danthi-ai-chatbot/
    danthi_ai_chatbot.py    ← main app file
    README.md               ← this file
    assets/
        owl.png             ← optional mascot image
```

---

## 🎯 How to Use Each Mode

**Chat Mode**
Type your question in the text box and press Send. Danthi remembers the full conversation.

**Voice Mode**
Press the 🎙️ START button, speak clearly, and wait. Danthi will reply in text and speak the answer aloud. Turn on Auto-loop to keep the conversation going without pressing anything.

**Web Scraper Mode**
Paste a full URL (example: `https://en.wikipedia.org/wiki/Python`) and press Scrape. Danthi summarises the page for you.

**Document Q&A Mode**
Upload a PDF file, then type your question. Danthi reads the document and finds the answer.

---

## ⚙️ System Requirements

- Windows 10 or 11
- Python 3.10
- At least 8GB RAM (16GB recommended)
- At least 10GB free disk space (for the AI model)
- Microphone (for voice mode)

---

## License

This project is open source and free to use for learning and personal projects.

---

## Acknowledgements

- [Ollama](https://ollama.com) — for making local LLMs easy
- [LangChain](https://langchain.com) — for the AI pipeline
- [Streamlit](https://streamlit.io) — for the beautiful UI
- Udemy — *Mastering AI Agents Bootcamp by School of AI*

---

> Made with 🌸 by Sahan Sandeepa Ellawala

