# =========================
# AI-Powered FAQ Chatbot
# Tech: FastAPI + NLTK + SQLite
# =========================

import nltk
import sqlite3
import json
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data (run once)
nltk.download('punkt')

# -------------------------
# Load FAQ Data
# -------------------------
faq_data = {
    "What is your return policy?": "You can return products within 30 days.",
    "How do I contact support?": "You can email us at support@example.com.",
    "What payment methods do you accept?": "We accept credit cards, debit cards, and UPI.",
    "Where are you located?": "We are located in Bangalore, India."
}

questions = list(faq_data.keys())
answers = list(faq_data.values())

# -------------------------
# NLP Vectorizer
# -------------------------
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

def get_best_response(user_input: str):
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, question_vectors)
    best_match_index = similarity.argmax()
    score = similarity[0][best_match_index]

    if score < 0.3:
        return "Sorry, I didn't understand that. Please try another question."
    return answers[best_match_index]

# -------------------------
# Database Setup
# -------------------------
conn = sqlite3.connect("chat_logs.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_message TEXT,
    bot_reply TEXT
)
""")
conn.commit()

def log_chat(user_message, bot_reply):
    cursor.execute(
        "INSERT INTO chats (user_message, bot_reply) VALUES (?, ?)",
        (user_message, bot_reply)
    )
    conn.commit()

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(title="AI Chatbot")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    reply = get_best_response(request.message)
    log_chat(request.message, reply)
    return {"reply": reply}

@app.get("/")
def home():
    return {"message": "Chatbot is running!"}
