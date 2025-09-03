from flask import Flask, render_template, request, jsonify
import cohere
import numpy as np
import os
from memory import FAISSMemory

# Flask app
app = Flask(__name__)

# Get API key securely
API_KEY = os.environ.get("COHERE_API_KEY", "gMMvQsKG4sRXRJWmkn83iJjsgd582ZcZLDJh17gh")

if not API_KEY:
    raise ValueError("Cohere API key not found. Set COHERE_API_KEY environment variable!")

co = cohere.Client(API_KEY)

# --- Helper: embed text ---
def embed_text(text):
    response = co.embed(
        texts=[text],
        model="embed-english-v3.0",
        input_type="search_query" 
    )
    return response.embeddings[0]

#  Cohere chat response ---
def generate_response_cohere_chat(user_query, retrieved_texts):
    context = "\n".join([f"- {role}: {txt}" for role, txt in retrieved_texts]) if retrieved_texts else "No prior context."

    message = f"""
    You are a compassionate and supportive mental health assistant.
    Always respond in a gentle, non-judgmental, and encouraging way.
    Keep your answers short (2–4 sentences max), supportive, and empathetic.
    Focus on validating the user’s feelings and offering one simple coping suggestion.

    Here are some past thoughts from the user:
    {context}

    Now, the user says: "{user_query}"
    Respond briefly, kindly, and helpfully.
    """

    response = co.chat(
        model="command-r-plus",
        message=message,
        temperature=0.7,
        max_tokens=120
    )
    return response.text.strip()

# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_query = data.get("message", "")
    user_id = data.get("user_id", "default_user")

    # 1. Load memory for this user
    memory = FAISSMemory(user_id, dim=1024)

    # 2. Embed query
    query_embedding = embed_text(user_query)

    # 3. Retrieve context
    retrieved = memory.search(query_embedding, top_k=5)

    # 4. Generate bot reply with context
    bot_reply = generate_response_cohere_chat(user_query, retrieved)

    # 5. Save user query + bot reply
    memory.add_memory(query_embedding, role="user", text=user_query)
    reply_embedding = embed_text(bot_reply)
    memory.add_memory(reply_embedding, role="bot", text=bot_reply)

    return jsonify({"reply": bot_reply})

# Debug route
@app.route("/show_memory/<user_id>", methods=["GET"])
def show_memory(user_id):
    memory = FAISSMemory(user_id, dim=1024)
    return jsonify({"user_id": user_id, "memories": memory.texts})

if __name__ == "__main__":
    print("Starting app with Cohere + FAISS memory")
    app.run(debug=True)
