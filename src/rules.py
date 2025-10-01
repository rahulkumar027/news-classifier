# src/rules.py

def load_sports_keywords(path="data/sports_keywords.txt"):
    """
    Load sports keywords/phrases from file.
    Each line in the file should contain one keyword or phrase.
    """
    try:
        with open(path, "r", encoding="utf8") as f:
            return [line.strip().lower() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"[WARN] sports_keywords.txt not found at {path}, using empty list.")
        return []


def rule_boost(headline, probs, classes, keywords, boost=0.3, threshold=0.55):
    """
    Adjust probabilities if sports keywords appear in the headline.
    - probs: numpy array of class probabilities
    - classes: list of class labels
    """
    text = headline.lower()
    matches = any(k in text for k in keywords)
    probs = probs.copy().astype(float)

    if matches:
        try:
            idx = list(classes).index("Sports")
            probs[idx] += boost
            probs /= probs.sum()  # normalize back
        except ValueError:
            pass

    pred_idx = probs.argmax()
    pred_label = classes[pred_idx]

    # Optional override
    if matches and pred_label != "Sports":
        sports_prob = probs[list(classes).index("Sports")]
        if sports_prob >= threshold:
            return "Sports", probs

    return pred_label, probs
