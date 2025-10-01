# src/utils.py
import joblib
from sklearn.metrics import classification_report, confusion_matrix

def save_model(obj, path):
    joblib.dump(obj, path)
    print(f"[INFO] Saved model to {path}")

def load_model(path):
    return joblib.load(path)

def evaluate_model(y_true, y_pred):
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
