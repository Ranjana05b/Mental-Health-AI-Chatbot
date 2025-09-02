from flask import Flask, render_template, request, jsonify
import cohere

app = Flask(__name__)

# Initialize Cohere client
co = cohere.Client("gMMvQsKG4sRXRJWmkn83iJjsgd582ZcZLDJh17gh")

def generate_response_cohere_chat(user_query, retrieved_texts):
    context = "\n".join(retrieved_texts)

    message = f"""
    You are a compassionate and supportive mental health assistant. 
    Always respond in a gentle, non-judgmental, and encouraging way. 
    Keep your answers short (2–4 sentences max), supportive, and empathetic. 
    Focus on validating the user’s feelings and offering one simple coping suggestion 
    (such as breathing, journaling, mindfulness, or a positive reminder). 

    Here are some related past thoughts from the user:
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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_query = data.get("message", "")
    retrieved_texts = ["How are you feeling?"]  # Replace with real RAG context if available
    
    bot_reply = generate_response_cohere_chat(user_query, retrieved_texts)
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
