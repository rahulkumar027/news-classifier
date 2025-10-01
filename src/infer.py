# src/infer.py
import numpy as np
from utils import load_model
from rules import load_sports_keywords, rule_boost

# ==== Load models ====
tfidf = load_model("models/tfidf_vec.joblib")
pipeline = load_model("models/pipeline_best.joblib")
calibrator = load_model("models/clf_calibrated.joblib")
classes = list(calibrator.classes_)

# ==== Load sports keywords ====
keywords = load_sports_keywords("data/sports_keywords.txt")

def predict_with_probs(headlines):
    """
    Get raw probabilities for one or more headlines.
    """
    Xv = tfidf.transform(headlines)
    probs = calibrator.predict_proba(Xv)
    return probs, classes

def predict_headline(headline):
    """
    Predict single headline with rule-based sports booster.
    Returns (label, probability_dict).
    """
    probs, classes = predict_with_probs([headline])
    probs = probs[0]  # shape (n_classes,)
    label, adjusted = rule_boost(headline, probs, classes, keywords,
                                 boost=0.35, threshold=0.55)
    return label, dict(zip(classes, adjusted.round(3)))


# ==== Example usage ====
if __name__ == "__main__":
    text = "Kapil Dev’s bold take on India-Pakistan Asia Cup drama: 'It’s time to…'"
    label, probs = predict_headline(text)
    print("Prediction:", label)
    print("Probabilities:", probs)
