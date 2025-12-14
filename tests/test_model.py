import joblib

def test_model_prediction():
    model = joblib.load("model/hate_speech_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

    text = ["I love everyone"]
    vec = vectorizer.transform(text)
    pred = model.predict(vec)

    assert pred[0] in [0, 1, 2]
