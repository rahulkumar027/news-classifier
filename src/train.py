# src/train.py
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from preprocessing import make_tfidf
from utils import save_model, evaluate_model

# ==== Load dataset ====
df = pd.read_csv("data/headlines.csv")  # must have 'text' and 'label' columns
X = df["text"].astype(str).values
y = df["label"].values

# ==== Build pipeline ====
tfidf = make_tfidf()
pipeline = Pipeline([
    ("tfidf", tfidf),
    ("clf", LogisticRegression(solver="saga", max_iter=2000,
                               class_weight="balanced", random_state=42))
])

# ==== Grid search ====
param_grid = {
    "tfidf__min_df": [1, 2],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.5, 1.0, 2.0]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipeline, param_grid, cv=cv,
                    scoring="f1_macro", n_jobs=-1, verbose=1)

print("[INFO] Training...")
grid.fit(X, y)
print("Best params:", grid.best_params_)

best_pipeline = grid.best_estimator_

# ==== Calibrate classifier ====
print("[INFO] Calibrating probabilities...")
calibrator = CalibratedClassifierCV(best_pipeline.named_steps["clf"],
                                    cv=3, method="sigmoid")
X_tfidf = best_pipeline.named_steps["tfidf"].transform(X)
calibrator.fit(X_tfidf, y)

# ==== Evaluate ====
y_pred = best_pipeline.predict(X)
evaluate_model(y, y_pred)

# ==== Save models ====
save_model(best_pipeline.named_steps["tfidf"], "models/tfidf_vec.joblib")
save_model(best_pipeline, "models/pipeline_best.joblib")
save_model(calibrator, "models/clf_calibrated.joblib")
