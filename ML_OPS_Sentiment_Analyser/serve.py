from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)
model = joblib.load("models/naive_bayes_sentiment.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
