import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
import logging
logging.getLogger("chromadb").setLevel(logging.ERROR)

import os
import uuid
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from dotenv import load_dotenv



load_dotenv()

# Suppress ChromaDB telemetry warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

import PyPDF2
import chromadb

# ─── Robust ChromaDB Embedding Import (works with all versions) ───────────────
try:
    from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
        SentenceTransformerEmbeddingFunction,
    )
except ImportError:
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError:
        from chromadb.utils import embedding_functions
        SentenceTransformerEmbeddingFunction = (
            embedding_functions.SentenceTransformerEmbeddingFunction
        )

from groq import Groq

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "documind-secret-key-change-this")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///ragapp.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

MAX_FILE_MB = 100
ALLOWED_EXTENSIONS = {"pdf", "txt"}

db = SQLAlchemy(app)

# ─── Database Models ──────────────────────────────────────────────────────────
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    chats = db.relationship("Chat", backref="user", lazy=True)


class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.String(36), unique=True, default=lambda: str(uuid.uuid4()))
    title = db.Column(db.String(200), default="New Chat")
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    messages = db.relationship("Message", backref="chat", lazy=True, cascade="all, delete-orphan")
    documents = db.relationship("Document", backref="chat", lazy=True, cascade="all, delete-orphan")


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey("chat.id"), nullable=False)
    role = db.Column(db.String(10), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey("chat.id"), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    collection_name = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ─── ChromaDB + Groq Setup ────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ─── Error Handlers ───────────────────────────────────────────────────────────
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({
        "error": f"Document size should be {MAX_FILE_MB} MB to upload files. "
                 f"Above {MAX_FILE_MB} MB, change the functionality or increase the document size."
    }), 413


# ─── Helper Functions ─────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def check_file_size(file):
    file.seek(0, 2)
    size_bytes = file.tell()
    file.seek(0)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb <= MAX_FILE_MB, size_mb


def extract_text(filepath, filename):
    ext = filename.rsplit(".", 1)[1].lower()
    if ext == "pdf":
        text = ""
        try:
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            raise ValueError(f"Could not read PDF: {str(e)}")
        return text
    elif ext == "txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def make_collection_name(chat_id, doc_id):
    # ChromaDB: 3-63 chars, alphanumeric + underscores only
    name = f"col_{chat_id[:8]}_{doc_id[:8]}".replace("-", "_")
    return name[:63]


def embed_document(text, collection_name, doc_id):
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No text chunks to embed.")
    collection = chroma_client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_fn
    )
    ids = [f"{doc_id}_c{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return len(chunks)


def search_context(query, collection_name, n_results=5):
    try:
        collection = chroma_client.get_collection(
            name=collection_name, embedding_function=embedding_fn
        )
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(query_texts=[query], n_results=min(n_results, count))
        return results["documents"][0] if results["documents"] else []
    except Exception:
        return []


def get_chat_collections(chat_db_id):
    docs = Document.query.filter_by(chat_id=chat_db_id).all()
    return [doc.collection_name for doc in docs]


def generate_answer(question, context_chunks, chat_history):
    if context_chunks:
        context = "\n\n---\n\n".join(context_chunks)
        system_prompt = (
            "You are a helpful AI assistant. Answer ONLY based on the provided document context.\n"
            "If the answer is found in the context, give a clear and accurate response.\n"
            "If the answer is NOT in the context, say: 'This information is not available in the uploaded document.'\n"
            "Do NOT use any outside knowledge. Only use what is in the document.\n"
            "Format your response in markdown (bullet points, code blocks, headers as needed).\n"
            "Be concise, clear, and helpful."
        )
        user_content = f"Document Context:\n{context}\n\nQuestion: {question}"
    else:
        system_prompt = (
            "You are a helpful AI assistant."
        )
        user_content = "Please tell the user: 'No document uploaded yet. Please upload a PDF or TXT file first, then ask your question.'"

    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_content})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # ← sirf yeh badlo
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
    )
    return response.choices[0].message.content


# ─── Auth Routes ──────────────────────────────────────────────────────────────
@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.json
    if not data or not all(k in data for k in ["email", "username", "password"]):
        return jsonify({"error": "All fields are required"}), 400
    if len(data["password"]) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    if User.query.filter_by(email=data["email"].lower()).first():
        return jsonify({"error": "Email already exists"}), 400
    if User.query.filter_by(username=data["username"]).first():
        return jsonify({"error": "Username already taken"}), 400
    user = User(
        username=data["username"].strip(),
        email=data["email"].strip().lower(),
        password=generate_password_hash(data["password"]),
    )
    db.session.add(user)
    db.session.commit()
    session["user_id"] = user.id
    session["username"] = user.username
    return jsonify({"message": "Account created!", "username": user.username})


@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    if not data or not data.get("email") or not data.get("password"):
        return jsonify({"error": "Email and password required"}), 400
    user = User.query.filter_by(email=data["email"].strip().lower()).first()
    if not user or not check_password_hash(user.password, data["password"]):
        return jsonify({"error": "Invalid email or password"}), 401
    session["user_id"] = user.id
    session["username"] = user.username
    return jsonify({"message": "Logged in!", "username": user.username})


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out"})


@app.route("/api/me")
def me():
    if "user_id" not in session:
        return jsonify({"logged_in": False})
    return jsonify({"logged_in": True, "username": session["username"]})


# ─── Chat Routes ──────────────────────────────────────────────────────────────
@app.route("/api/chats", methods=["GET"])
def get_chats():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    chats = Chat.query.filter_by(user_id=session["user_id"]).order_by(Chat.created_at.desc()).all()
    return jsonify([
        {"chat_id": c.chat_id, "title": c.title, "created_at": c.created_at.isoformat()}
        for c in chats
    ])


@app.route("/api/chats", methods=["POST"])
def create_chat():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    chat = Chat(user_id=session["user_id"], title="New Chat")
    db.session.add(chat)
    db.session.commit()
    return jsonify({"chat_id": chat.chat_id, "title": chat.title})


@app.route("/api/chats/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    chat = Chat.query.filter_by(chat_id=chat_id, user_id=session["user_id"]).first_or_404()
    for doc in chat.documents:
        try:
            chroma_client.delete_collection(doc.collection_name)
        except Exception:
            pass
    db.session.delete(chat)
    db.session.commit()
    return jsonify({"message": "Chat deleted"})


@app.route("/api/chats/<chat_id>/messages", methods=["GET"])
def get_messages(chat_id):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    chat = Chat.query.filter_by(chat_id=chat_id, user_id=session["user_id"]).first_or_404()
    messages = Message.query.filter_by(chat_id=chat.id).order_by(Message.created_at).all()
    docs = Document.query.filter_by(chat_id=chat.id).all()
    return jsonify({
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "documents": [{"filename": d.filename} for d in docs],
    })


# ─── Upload Route ─────────────────────────────────────────────────────────────
@app.route("/api/upload/<chat_id>", methods=["POST"])
def upload_file(chat_id):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    chat = Chat.query.filter_by(chat_id=chat_id, user_id=session["user_id"]).first_or_404()

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF and TXT files are allowed"}), 400

    # Check 100MB size limit
    size_ok, size_mb = check_file_size(file)
    if not size_ok:
        return jsonify({
            "error": f"Document size should be {MAX_FILE_MB} MB to upload files. "
                     f"Above {MAX_FILE_MB} MB, change the functionality or increase the document size. "
                     f"(Your file: {size_mb:.1f} MB)"
        }), 413

    filename = secure_filename(file.filename)
    doc_id = str(uuid.uuid4())[:8]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{doc_id}_{filename}")

    try:
        file.save(filepath)
        text = extract_text(filepath, filename)
        if not text.strip():
            return jsonify({
                "error": "Could not extract text. Make sure the PDF is not a scanned image."
            }), 400

        collection_name = make_collection_name(chat_id, doc_id)
        chunks = embed_document(text, collection_name, doc_id)

        doc = Document(chat_id=chat.id, filename=filename, collection_name=collection_name)
        db.session.add(doc)

        if chat.title == "New Chat":
            chat.title = filename[:40]
        db.session.commit()

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return jsonify({
        "message": f"Successfully uploaded '{filename}' — {chunks} chunks embedded.",
        "filename": filename,
        "chunks": chunks,
        "size_mb": round(size_mb, 2),
    })


# ─── Ask Route ────────────────────────────────────────────────────────────────
@app.route("/api/ask/<chat_id>", methods=["POST"])
def ask(chat_id):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    chat = Chat.query.filter_by(chat_id=chat_id, user_id=session["user_id"]).first_or_404()
    data = request.json
    if not data:
        return jsonify({"error": "Invalid request"}), 400

    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    collection_names = get_chat_collections(chat.id)
    context_chunks = []
    for col_name in collection_names:
        context_chunks.extend(search_context(question, col_name))

    msgs = Message.query.filter_by(chat_id=chat.id).order_by(Message.created_at).all()
    history = [{"role": m.role, "content": m.content} for m in msgs]

    try:
        answer = generate_answer(question, context_chunks, history)
    except Exception as e:
        return jsonify({"error": f"AI error: {str(e)}"}), 500

    db.session.add(Message(chat_id=chat.id, role="user", content=question))
    db.session.add(Message(chat_id=chat.id, role="assistant", content=answer))

    if chat.title == "New Chat" and len(msgs) == 0:
        chat.title = question[:50]
    db.session.commit()

    return jsonify({"answer": answer})


# ─── Main Route ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    with app.app_context():
        db.create_all()
    print("\n🚀 DocuMind AI is running at: http://localhost:5000\n")
    app.run(debug=True, port=5000)
