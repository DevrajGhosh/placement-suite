"""
=============================================================
  PHASE 2 — TRAIN ALL ML MODELS  (UPGRADED VERSION)
  AI Placement Suite | College Project
=============================================================
New in this version:
  - Model 1: Compare RF vs XGBoost vs SVM vs Logistic Regression
    → Best model auto-selected and saved
  - Model 2: Salary tier binary classifier (fixed, working)
  - Model 3: Resume TF-IDF scorer (unchanged)
  - Saves comparison_results.json for frontend chart
=============================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

os.makedirs("backend/models", exist_ok=True)

df = pd.read_csv("data/processed_placement.csv")
print(f"Dataset: {df.shape[0]} records | Placed: {df['placed'].sum()} | Not placed: {(df['placed']==0).sum()}\n")

FEAT = ["gender","ssc_p","hsc_p","degree_p","workex",
        "etest_p","specialisation","mba_p","academic_score"]
X = df[FEAT]; y = df["placed"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL COMPARISON: 6 algorithms side by side
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  MODEL COMPARISON — Placement Predictor")
print("=" * 60)

MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, verbosity=0),
    "SVM": SVC(
        C=1.0, kernel="rbf", probability=True,
        class_weight="balanced", random_state=42),
    "Logistic Regression": LogisticRegression(
        C=1.0, class_weight="balanced",
        max_iter=1000, random_state=42),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=7),
}

comparison_results = []
best_model_name = None
best_cv_score   = 0
best_pipeline   = None

print(f"\n{'Model':<22} {'Test Acc':>9} {'CV Acc':>9} {'CV Std':>8} {'AUC':>8}")
print("-" * 60)

for name, clf in MODELS.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_pred)
    auc      = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    cv_mean   = cv_scores.mean()
    cv_std    = cv_scores.std()

    print(f"{name:<22} {test_acc*100:>8.1f}% {cv_mean*100:>8.1f}% {cv_std*100:>7.1f}% {auc:>8.3f}")

    comparison_results.append({
        "model":    name,
        "test_acc": round(test_acc * 100, 1),
        "cv_acc":   round(cv_mean * 100, 1),
        "cv_std":   round(cv_std * 100, 1),
        "auc":      round(auc, 3),
    })

    if cv_mean > best_cv_score:
        best_cv_score   = cv_mean
        best_model_name = name
        best_pipeline   = pipe

print("-" * 60)
print(f"\n🏆 Best model: {best_model_name} (CV: {best_cv_score*100:.1f}%)")

# Detailed report for best model
y_pred_best = best_pipeline.predict(X_test)
print(f"\nDetailed report — {best_model_name}:")
print(classification_report(y_test, y_pred_best))

# Feature importance (RF or XGBoost only)
if best_model_name in ("Random Forest", "XGBoost"):
    fi = pd.Series(
        best_pipeline.named_steps["clf"].feature_importances_,
        index=FEAT).sort_values(ascending=False)
    print("Feature Importance:")
    for feat, imp in fi.items():
        print(f"  {feat:<20} {imp:.4f}  {'█'*int(imp*40)}")
    comparison_results.append({
        "feature_importance": {k: round(float(v), 4) for k, v in fi.items()}
    })

# Save comparison JSON for frontend chart
with open("backend/models/comparison_results.json", "w") as f:
    json.dump(comparison_results, f, indent=2)

joblib.dump(best_pipeline, "backend/models/placement_model.pkl")
joblib.dump(FEAT,          "backend/models/placement_features.pkl")
joblib.dump(best_model_name, "backend/models/best_model_name.pkl")
print(f"\n✅ Saved best model: {best_model_name}")
print("✅ Saved comparison_results.json\n")


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL 2: SALARY TIER CLASSIFIER (binary, above/below median)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  MODEL 2: Salary Tier Classifier")
print("=" * 60)

df_placed  = df[df["placed"] == 1].copy()
median_sal = df_placed["salary"].median()
sal_min    = df_placed["salary"].min()
sal_max    = df_placed["salary"].max()

df_placed["salary_tier"] = (df_placed["salary"] > median_sal).astype(int)

FEAT_SAL  = ["gender","ssc_p","hsc_p","degree_p","workex",
             "etest_p","specialisation","mba_p","academic_score"]
X_sal = df_placed[FEAT_SAL]; y_sal = df_placed["salary_tier"]
X_str, X_ste, y_str, y_ste = train_test_split(
    X_sal, y_sal, test_size=0.2, random_state=42, stratify=y_sal)

# Compare salary models too
SAL_MODELS = {
    "Random Forest":      RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42),
    "XGBoost":            xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, verbosity=0, random_state=42),
    "Logistic Regression":LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42),
    "SVM":                SVC(C=1.0, probability=True, class_weight="balanced", random_state=42),
}

sal_results = []
best_sal_name  = None
best_sal_score = 0
best_sal_pipe  = None

print(f"\n{'Model':<22} {'Test Acc':>9} {'CV Acc':>9}")
print("-" * 44)
for name, clf in SAL_MODELS.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_str, y_str)
    test_acc  = accuracy_score(y_ste, pipe.predict(X_ste))
    cv_mean   = cross_val_score(pipe, X_sal, y_sal, cv=5, scoring="accuracy").mean()
    print(f"{name:<22} {test_acc*100:>8.1f}% {cv_mean*100:>8.1f}%")
    sal_results.append({"model": name, "test_acc": round(test_acc*100,1), "cv_acc": round(cv_mean*100,1)})
    if cv_mean > best_sal_score:
        best_sal_score = cv_mean; best_sal_name = name; best_sal_pipe = pipe

print(f"\n🏆 Best salary model: {best_sal_name} (CV: {best_sal_score*100:.1f}%)")

SALARY_META = {
    "model_type":   "binary_tier",
    "best_model":   best_sal_name,
    "median":       float(median_sal),
    "sal_min":      float(sal_min),
    "sal_max":      float(sal_max),
    "tier_names":   ["Low", "High"],
    "tier_ranges":  {
        "Low":  f"₹{sal_min/1e5:.1f}L – ₹{median_sal/1e5:.1f}L",
        "High": f"₹{(median_sal+1)/1e5:.1f}L – ₹{sal_max/1e5:.1f}L",
    },
    "tier_midpoints": {
        "Low":  int((sal_min + median_sal) / 2),
        "High": int((median_sal + sal_max) / 2),
    },
    "tier_notes": {
        "Low":  "Below-median package. Improve MBA % and entrance test score to move up.",
        "High": "Above-median package. Strong academic and experience profile.",
    },
    "salary_comparison": sal_results,
}

joblib.dump(best_sal_pipe, "backend/models/salary_model.pkl")
joblib.dump(FEAT_SAL,      "backend/models/salary_features.pkl")
joblib.dump(SALARY_META,   "backend/models/salary_meta.pkl")
for old in ["salary_log_transform.pkl","salary_model_type.pkl",
            "salary_band_ranges.pkl","salary_band_labels.pkl"]:
    p = f"backend/models/{old}"
    if os.path.exists(p): os.remove(p)
print(f"✅ Saved salary model: {best_sal_name}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL 3: RESUME SCORER (TF-IDF)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  MODEL 3: Resume Scorer (TF-IDF)")
print("=" * 60)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KEYWORD_BANK = {
    "software_engineer": [
        "python","java","c++","javascript","react","node","django","flask",
        "sql","nosql","mongodb","docker","kubernetes","aws","git","api",
        "rest","microservices","agile","data structures","algorithms","oop",
        "linux","system design","typescript","redis","ci cd","testing","html","css","spring"
    ],
    "data_scientist": [
        "python","r","machine learning","deep learning","tensorflow","pytorch",
        "scikit-learn","pandas","numpy","sql","statistics","nlp","computer vision",
        "tableau","power bi","regression","classification","clustering",
        "neural network","feature engineering","model deployment","spark",
        "hypothesis testing","data visualization","a/b testing","jupyter"
    ],
    "marketing_mba": [
        "marketing","branding","digital marketing","seo","sem","social media",
        "campaign","analytics","consumer behaviour","market research","crm",
        "salesforce","b2b","b2c","product management","strategy","excel",
        "powerpoint","leadership","segmentation","roi","kpi","brand equity",
        "content marketing","email marketing","google analytics"
    ],
    "finance_mba": [
        "financial analysis","valuation","excel","financial modelling",
        "investment","equity","derivatives","risk management","bloomberg",
        "accounting","audit","taxation","balance sheet","cash flow",
        "portfolio","trading","banking","fintech","dcf","npv",
        "irr","wacc","mergers","acquisitions","credit analysis"
    ]
}

corpus     = [" ".join(kws) for kws in KEYWORD_BANK.values()]
categories = list(KEYWORD_BANK.keys())
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
tfidf_mat  = vectorizer.fit_transform(corpus)

sample = "Python machine learning scikit-learn pandas numpy tensorflow sql jupyter"
rv   = vectorizer.transform([sample.lower()])
sims = cosine_similarity(rv, tfidf_mat)[0]
best = categories[sims.argmax()]
kws  = KEYWORD_BANK[best]
pres = [k for k in kws if k in sample.lower()]
print(f"Sample score: {round(len(pres)/len(kws)*100,1)}% | Best match: {best}")

joblib.dump((vectorizer, tfidf_mat, categories, KEYWORD_BANK),
            "backend/models/resume_scorer.pkl")
print("✅ Saved resume_scorer.pkl\n")


# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("  All models trained!")
print("=" * 60)
print(f"\n  Placement — Best: {best_model_name}  CV: {best_cv_score*100:.1f}%")
print(f"  Salary    — Best: {best_sal_name}  CV: {best_sal_score*100:.1f}%")
print(f"  Resume    — TF-IDF cosine similarity: active")
print(f"\n  Files saved:")
for fname in sorted(os.listdir("backend/models")):
    size = os.path.getsize(f"backend/models/{fname}")
    print(f"    {fname:<42} {size/1024:.1f} KB")