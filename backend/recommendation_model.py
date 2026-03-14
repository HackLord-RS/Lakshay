"""
LAKSHAY — Career Recommendation ML Model
==========================================
Trains a three-model ensemble and predicts top-N career matches.

Architecture
------------
  RandomForestClassifier   (weight 0.50) — robust, handles non-linear boundaries
  KNeighborsClassifier     (weight 0.30) — instance-based, good for rare profiles
  DecisionTreeClassifier   (weight 0.20) — interpretable, adds ensemble diversity

All models use a StandardScaler pipeline.
Final prediction = weighted average of probability vectors.

Features (7):  analytical, technical, creative, social, leadership, business, research
Target:        career_id  (20 unique classes)
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble      import RandomForestClassifier
from sklearn.tree          import DecisionTreeClassifier
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline      import Pipeline
from sklearn.model_selection import cross_val_score

DATA_DIR  = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "lakshay_model.pkl"

FEATURE_COLS = ["analytical","technical","creative","social","leadership","business","research"]
TARGET_COL   = "career_id"
LABEL_COL    = "career_label"

# Ensemble weights
RF_W  = 0.50
KNN_W = 0.30
DT_W  = 0.20

MODEL_DIR.mkdir(exist_ok=True)


# ── Training ────────────────────────────────────────────────────────────

def train(force: bool = False) -> dict:
    """
    Train (or reload) the career recommendation ensemble.

    Parameters
    ----------
    force : bool  If True, retrain even if a saved model exists.

    Returns
    -------
    dict  {status, rf_cv_acc, knn_cv_acc, dt_cv_acc, n_samples, n_classes}
    """
    if MODEL_PATH.exists() and not force:
        return {"status": "loaded_from_cache", **_get_cached_metrics()}

    df = pd.read_csv(DATA_DIR / "training_data.csv")
    X  = df[FEATURE_COLS].values.astype(float)

    le   = LabelEncoder()
    y    = le.fit_transform(df[TARGET_COL].values)

    # Build label→display map
    id_to_label = dict(zip(df[TARGET_COL].values, df[LABEL_COL].values))

    # ── Pipelines ──────────────────────────────────────────────────────
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200, max_depth=None,
            min_samples_split=2, random_state=42,
            class_weight="balanced"
        )),
    ])
    knn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    KNeighborsClassifier(
            n_neighbors=5, weights="distance", metric="euclidean"
        )),
    ])
    dt_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    DecisionTreeClassifier(
            max_depth=12, random_state=42, class_weight="balanced"
        )),
    ])

    rf_pipe.fit(X, y)
    knn_pipe.fit(X, y)
    dt_pipe.fit(X, y)

    rf_cv  = float(cross_val_score(rf_pipe,  X, y, cv=5, scoring="accuracy").mean())
    knn_cv = float(cross_val_score(knn_pipe, X, y, cv=5, scoring="accuracy").mean())
    dt_cv  = float(cross_val_score(dt_pipe,  X, y, cv=5, scoring="accuracy").mean())

    bundle = {
        "rf_pipe":       rf_pipe,
        "knn_pipe":      knn_pipe,
        "dt_pipe":       dt_pipe,
        "label_encoder": le,
        "id_to_label":   id_to_label,
        "feature_cols":  FEATURE_COLS,
        "metrics": {
            "rf_cv_acc":  round(rf_cv,  4),
            "knn_cv_acc": round(knn_cv, 4),
            "dt_cv_acc":  round(dt_cv,  4),
            "n_samples":  len(X),
            "n_classes":  len(le.classes_),
        }
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    return {
        "status":     "trained",
        "rf_cv_acc":  round(rf_cv,  4),
        "knn_cv_acc": round(knn_cv, 4),
        "dt_cv_acc":  round(dt_cv,  4),
        "n_samples":  len(X),
        "n_classes":  len(le.classes_),
    }


def _get_cached_metrics() -> dict:
    bundle = _load()
    return bundle.get("metrics", {})


def _load() -> dict:
    if not MODEL_PATH.exists():
        train()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ── Prediction ──────────────────────────────────────────────────────────

def predict(normalized_scores: dict, top_n: int = 5) -> list:
    """
    Predict top-N career recommendations.

    Parameters
    ----------
    normalized_scores : dict  {dimension: int 0-100}
    top_n             : int   Number of careers to return.

    Returns
    -------
    list of dicts:
        [{rank, career_id, career_name, match_score (0-100), confidence_label}]
    """
    bundle = _load()
    le, rf, knn, dt = (
        bundle["label_encoder"], bundle["rf_pipe"],
        bundle["knn_pipe"],      bundle["dt_pipe"]
    )
    id2lbl = bundle["id_to_label"]

    X = np.array([[normalized_scores.get(f, 50) for f in FEATURE_COLS]])

    rf_prob  = rf.predict_proba(X)[0]
    knn_prob = knn.predict_proba(X)[0]
    dt_prob  = dt.predict_proba(X)[0]

    ensemble = RF_W * rf_prob + KNN_W * knn_prob + DT_W * dt_prob

    classes = le.classes_
    ranked  = sorted(zip(classes, ensemble), key=lambda x: x[1], reverse=True)

    results = []
    for rank, (career_id, prob) in enumerate(ranked[:top_n], 1):
        # Scale probability to human-readable match score
        # Rank-1 scaled between 72–98; others proportionally lower
        raw_score = int(round(prob * 100 * 1.3))
        raw_score = min(raw_score, 98)
        if rank == 1:
            raw_score = max(raw_score, 72)
        raw_score = max(raw_score, 10)

        confidence = (
            "Very High" if raw_score >= 80 else
            "High"      if raw_score >= 65 else
            "Moderate"  if raw_score >= 48 else
            "Low"
        )

        results.append({
            "rank":         rank,
            "career_id":    career_id,
            "career_name":  id2lbl.get(career_id, career_id.replace("_", " ").title()),
            "match_score":  raw_score,
            "confidence":   confidence,
            "raw_prob":     round(float(prob), 5),
        })

    return results


def model_status() -> dict:
    """Return whether model is trained and accuracy metrics."""
    if MODEL_PATH.exists():
        m = _get_cached_metrics()
        return {"trained": True, **m}
    return {"trained": False}


# ── Stream & education path lookup ─────────────────────────────────────

_CAREER_STREAM = {
    "data_scientist":         ("Science – Technology",          ["Class 11–12: PCM + CS", "B.Tech / B.Sc CS or Statistics"]),
    "ai_ml_engineer":         ("Science – Technology",          ["Class 11–12: PCM + CS", "B.Tech CS / AI"]),
    "software_engineer":      ("Science – Technology",          ["Class 11–12: PCM + CS", "B.Tech CS or IT"]),
    "robotics_engineer":      ("Science – Engineering",         ["Class 11–12: PCM", "B.Tech Mech / Mechatronics / ECE"]),
    "research_scientist":     ("Science – Research",            ["Class 11–12: PCM or PCB", "B.Sc then M.Sc + Ph.D"]),
    "biomedical_engineer":    ("Science – Medical Technology",  ["Class 11–12: PCB + Maths", "B.Tech Biomedical Engineering"]),
    "ux_ui_designer":         ("Arts – Design Technology",      ["Class 11–12: Any stream", "B.Des from NID / UCEED"]),
    "clinical_psychologist":  ("Arts / Science – Psychology",   ["Class 11–12: PCB or Humanities", "B.Sc/BA Psychology → M.Phil RCI"]),
    "chartered_accountant":   ("Commerce – Finance",            ["Class 11–12: Commerce", "CA Foundation → Intermediate → Final"]),
    "investment_banker":      ("Commerce – Finance",            ["Class 11–12: Commerce or PCM", "BBA/B.Tech + MBA Finance"]),
    "entrepreneur":           ("Any Stream – Entrepreneurship", ["Any stream", "Any degree + Incubation or self-learning"]),
    "civil_services":         ("Any Stream – Civil Services",   ["Any stream", "Any graduation + UPSC CSE"]),
    "architect":              ("Science / Arts – Architecture", ["Class 11–12: PCM (Maths essential)", "B.Arch 5 years via NATA"]),
    "lawyer":                 ("Humanities / Any – Law",        ["Class 11–12: Any stream", "5-yr LLB via CLAT"]),
    "doctor_mbbs":            ("Science – Medical",             ["Class 11–12: PCB mandatory", "MBBS 5.5 years via NEET-UG"]),
    "product_manager":        ("Any Stream – Product Mgmt",     ["Any graduation", "MBA / lateral transition from tech or design"]),
    "fashion_designer":       ("Arts – Fashion Design",         ["Class 11–12: Any (Arts preferred)", "B.Des NIFT / NID"]),
    "journalist":             ("Humanities – Journalism",       ["Class 11–12: Any stream", "BA Journalism & Mass Communication"]),
    "environmental_scientist":("Science – Environment",         ["Class 11–12: PCB or PCM", "B.Sc/B.Tech Environmental Science"]),
    "teacher_educator":       ("Any Stream – Education",        ["Any stream", "B.Sc/BA + B.Ed then UGC-NET"]),
}


def get_stream(career_id: str) -> tuple:
    """Return (stream_name, education_steps[]) for a career id."""
    return _CAREER_STREAM.get(
        career_id,
        ("Science – Technology", ["PCM + CS", "B.Tech CS"])
    )