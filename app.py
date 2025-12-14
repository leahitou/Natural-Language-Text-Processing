from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Loads model and vectorizer
model = joblib.load("model/hate_speech_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

labels = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither"
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        text = request.form["text"]
        vector = vectorizer.transform([text])
        pred = model.predict(vector)[0]
        prediction = labels[pred]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
