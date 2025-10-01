# app.py
import os
from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

MODEL_PATH = "model/news_classifier.joblib"
app = Flask(__name__)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run `python train_model.py` first to create it.")

model = joblib.load(MODEL_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or request.form
    text = data.get("text", "")
    if not text or not text.strip():
        return jsonify({"error": "No text provided"}), 400

    pred = model.predict([text])[0]

    # probabilities
    prob_dict = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        prob_dict = dict(zip(model.classes_.tolist(), probs.tolist()))
    elif hasattr(model, "decision_function"):
        scores = model.decision_function([text])[0]
        # softmax
        exps = np.exp(scores - np.max(scores))
        probs = exps / exps.sum()
        prob_dict = dict(zip(model.classes_.tolist(), probs.tolist()))
    else:
        prob_dict = {pred: 1.0}

    # simple explanation using coefficients (works for linear models)
    explanation = {}
    try:
        if hasattr(model, "named_steps"):
            tf = model.named_steps["tfidf"]
            clf = model.named_steps["clf"]
            if hasattr(clf, "coef_"):
                tokens = tf.get_feature_names_out()
                class_index = list(model.classes_).index(pred)
                coefs = clf.coef_[class_index]
                top_idx = np.argsort(coefs)[-10:][::-1]
                top = [{"token": tokens[i], "coef": float(coefs[i])} for i in top_idx]
                explanation["top_tokens"] = top
    except Exception:
        explanation = {}

    return jsonify({"prediction": pred, "probabilities": prob_dict, "explanation": explanation})

if __name__ == "__main__":
    # dev server
    app.run(host="0.0.0.0", port=5000, debug=True)
