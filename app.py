from transformers import pipeline
from flask import Flask, request, jsonify

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Sentiment Analysis API is running!"

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Perform sentiment analysis
    result = sentiment_pipeline(text)[0]
    return jsonify({
        "text": text,
        "sentiment": result["label"],
        "confidence": round(result["score"], 4)
    })

if __name__ == "__main__":
    print("Starting Sentiment Analysis API...")
    app.run(debug=True)
