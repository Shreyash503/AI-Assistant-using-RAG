# 🤖 DOCS — RAG Chatbot

Chat with your PDF/TXT documents using AI (Groq + ChromaDB + Flask)

---

## 📁 Project Structure

```
rag_project/
├── app.py              ← Main Flask app (Backend + RAG pipeline)
├── requirements.txt    ← Python dependencies
├── .env                ← Your API keys (create from .env.example)
├── .env.example        ← Template for .env
├── templates/
│   └── index.html      ← ChatGPT-like frontend UI
├── uploads/            ← Temp file uploads (auto-created)
├── chroma_db/          ← Vector database (auto-created)
└── instance/
    └── ragapp.db       ← SQLite database (auto-created)
```

---

## ⚡ Setup Steps

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Groq API Key (FREE)
- Go to: https://console.groq.com
- Sign up and create API key
- It's FREE with generous limits

### 3. Create .env file
```bash
cp .env.example .env
```
Then edit `.env` and add your Groq API key:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
SECRET_KEY=any-random-secret-string-here
```

### 4. Run the app
```bash
python app.py
```

### 5. Open browser
```
http://localhost:5000
```

---

## 🔧 How It Works (RAG Pipeline)

```
User uploads PDF
      ↓
Extract text (PyPDF2)
      ↓
Split into chunks (500 words, 50 overlap)
      ↓
Generate embeddings (sentence-transformers)
      ↓
Store in ChromaDB (vector database)
      ↓
User asks question
      ↓
Search similar chunks (cosine similarity)
      ↓
Send context + question to Groq LLM
      ↓
Stream answer back to user
```

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask (Python) |
| LLM | Groq API (Llama3-8b) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| Database | SQLite (via SQLAlchemy) |
| PDF Parsing | PyPDF2 |
| Frontend | HTML + CSS + JS (ChatGPT-like UI) |

---

## 📋 Features

- ✅ User Signup / Login / Logout
- ✅ Upload PDF or TXT files
- ✅ RAG-based question answering
- ✅ Chat history saved per user
- ✅ Multiple chats support
- ✅ New chat creation
- ✅ Delete chats
- ✅ Guest mode (no login required, no save)
- ✅ Markdown rendering in responses
- ✅ Code syntax highlighting
- ✅ Typing indicator animation

---

## ❓ Troubleshooting

**"Could not extract text"** → PDF might be scanned/image-based. Use a text-based PDF.

**Groq API error** → Check your API key in `.env` file.

**chromadb error on first run** → Run `pip install chromadb --upgrade`

**sentence-transformers slow on first run** → It downloads the model (~90MB) once. Be patient!
