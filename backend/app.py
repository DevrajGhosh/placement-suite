"""
=============================================================
  PHASE 3 — FLASK REST API BACKEND
  AI Placement Suite | College Project
=============================================================
Run:
  python app.py   (dev)
  gunicorn app:app  (prod)

Endpoints:
  POST /api/auth/register
  POST /api/auth/login
  POST /api/predict/placement
  POST /api/predict/salary
  POST /api/resume/score
  POST /api/interview/chat
  GET  /api/history/<user_id>
=============================================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import (JWTManager, create_access_token,
                                 jwt_required, get_jwt_identity)
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
import joblib
import numpy as np
import pandas as pd
import pdfplumber
import os
import io
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "dev-secret-change-in-prod")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)
jwt = JWTManager(app)

# ── Database ──────────────────────────────────────────────────────────────────
engine = create_engine("sqlite:///placement.db", echo=False)
Base = declarative_base()
Session = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = "users"
    id       = Column(Integer, primary_key=True)
    name     = Column(String(100))
    email    = Column(String(120), unique=True)
    password = Column(String(256))
    created  = Column(DateTime, default=datetime.utcnow)

class Prediction(Base):
    __tablename__ = "predictions"
    id         = Column(Integer, primary_key=True)
    user_id    = Column(Integer)
    type       = Column(String(50))   # 'placement' | 'salary' | 'resume'
    input_data = Column(Text)
    result     = Column(Text)
    timestamp  = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# ── Load ML Models ────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models")

try:
    placement_model   = joblib.load(os.path.join(MODEL_PATH, "placement_model.pkl"))
    placement_feats   = joblib.load(os.path.join(MODEL_PATH, "placement_features.pkl"))
    salary_model      = joblib.load(os.path.join(MODEL_PATH, "salary_model.pkl"))
    salary_feats      = joblib.load(os.path.join(MODEL_PATH, "salary_features.pkl"))
    resume_scorer_pkg = joblib.load(os.path.join(MODEL_PATH, "resume_scorer.pkl"))
    vectorizer, tfidf_matrix, categories, KEYWORD_BANK = resume_scorer_pkg
    # Salary metadata (works with all model versions)
    _meta_path = os.path.join(MODEL_PATH, "salary_meta.pkl")
    salary_meta = joblib.load(_meta_path) if os.path.exists(_meta_path) else {}
    print(f"✅ All models loaded. Salary model: {salary_meta.get('model_type','unknown')}")
except FileNotFoundError as e:
    print(f"⚠️  Model not found: {e}. Run 02_train_models.py first.")
    placement_model = salary_model = None
    salary_meta = {}

# ── Helper ────────────────────────────────────────────────────────────────────
def _compute_academic_score(d):
    return (0.25*d["ssc_p"] + 0.25*d["hsc_p"] +
            0.30*d["degree_p"] + 0.20*d["etest_p"])

def score_resume_text(resume_text: str, target_role: str = None) -> dict:
    from sklearn.metrics.pairwise import cosine_similarity
    resume_lower = resume_text.lower()
    resume_vec = vectorizer.transform([resume_lower])
    sims = cosine_similarity(resume_vec, tfidf_matrix)[0]
    role_scores = {cat: round(float(s)*100, 1) for cat, s in zip(categories, sims)}
    best_match  = max(role_scores, key=role_scores.get)
    kws = KEYWORD_BANK.get(target_role or best_match, [])
    present = [k for k in kws if k in resume_lower]
    missing = [k for k in kws if k not in resume_lower]
    return {
        "overall_score": round(len(present)/len(kws)*100, 1) if kws else 0,
        "best_match": best_match,
        "role_scores": role_scores,
        "present_keywords": present,
        "missing_keywords": missing[:10]
    }

# ─────────────────────────────────────────────────────────────────────────────
#  AUTH ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.json
    db   = Session()
    if db.query(User).filter_by(email=data["email"]).first():
        return jsonify({"error": "Email already registered"}), 409
    user = User(
        name     = data["name"],
        email    = data["email"],
        password = generate_password_hash(data["password"])
    )
    db.add(user); db.commit()
    token = create_access_token(identity=str(user.id))
    return jsonify({"token": token, "user": {"id": user.id, "name": user.name}}), 201

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.json
    db   = Session()
    user = db.query(User).filter_by(email=data["email"]).first()
    if not user or not check_password_hash(user.password, data["password"]):
        return jsonify({"error": "Invalid credentials"}), 401
    token = create_access_token(identity=str(user.id))
    return jsonify({"token": token, "user": {"id": user.id, "name": user.name}}), 200

# ─────────────────────────────────────────────────────────────────────────────
#  PLACEMENT PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/predict/placement", methods=["POST"])
@jwt_required()
def predict_placement():
    """
    Input JSON:
    {
      "gender": 0|1,           # 0=F, 1=M
      "ssc_p": 75.0,           # 10th % 
      "hsc_p": 70.0,           # 12th %
      "degree_p": 68.0,        # UG %
      "workex": 0|1,           # work experience
      "etest_p": 72.0,         # entrance test %
      "specialisation": 0|1,   # 0=Mkt&HR, 1=Mkt&Finance
      "mba_p": 65.0            # MBA %
    }
    """
    if placement_model is None:
        return jsonify({"error": "Model not trained. Run 02_train_models.py"}), 503

    d = request.json
    d["academic_score"] = _compute_academic_score(d)

    X = pd.DataFrame([[d[f] for f in placement_feats]], columns=placement_feats)
    prob  = float(placement_model.predict_proba(X)[0][1])
    pred  = int(placement_model.predict(X)[0])

    # Advice
    advice = []
    if d["academic_score"] < 65:
        advice.append("Focus on improving your academic performance.")
    if d["workex"] == 0:
        advice.append("Gain internship or project experience.")
    if d["mba_p"] < 65:
        advice.append("Strengthen MBA coursework and case study skills.")
    if not advice:
        advice.append("You have a strong profile. Keep it up!")

    result = {
        "placed": pred == 1,
        "probability": round(prob * 100, 1),
        "confidence": "High" if prob > 0.75 else "Medium" if prob > 0.5 else "Low",
        "advice": advice
    }

    # Save to DB
    uid = get_jwt_identity()
    db  = Session()
    db.add(Prediction(user_id=uid, type="placement",
                      input_data=str(d), result=str(result)))
    db.commit()

    return jsonify(result), 200

# ─────────────────────────────────────────────────────────────────────────────
#  SALARY PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/predict/salary", methods=["POST"])
@jwt_required()
def predict_salary():
    """
    Input JSON (same as placement predictor):
    { "gender":0, "ssc_p":75, "hsc_p":70, "degree_p":68,
      "workex":1, "etest_p":74, "specialisation":1, "mba_p":65 }
    """
    if salary_model is None:
        return jsonify({"error": "Model not trained. Run 02_train_models.py"}), 503

    d = request.json
    d["academic_score"] = _compute_academic_score(d)
    X = pd.DataFrame([[d[f] for f in salary_feats]], columns=salary_feats)

    # Predict tier (0=Low, 1=High) and get probabilities
    tier_idx    = int(salary_model.predict(X)[0])
    tier_proba  = salary_model.predict_proba(X)[0]
    confidence  = round(float(tier_proba[tier_idx]) * 100, 1)

    # Look up range and description from saved metadata
    tier_names    = salary_meta.get("tier_names",    ["Low", "High"])
    tier_ranges   = salary_meta.get("tier_ranges",   {"Low": "₹2L–₹2.65L", "High": "₹2.65L–₹9.4L"})
    tier_midpoints= salary_meta.get("tier_midpoints",{"Low": 232500, "High": 602500})
    tier_notes    = salary_meta.get("tier_notes",    {"Low": "Below median", "High": "Above median"})

    tier_name     = tier_names[tier_idx] if tier_idx < len(tier_names) else "Unknown"
    midpoint      = tier_midpoints.get(tier_name, 265000)
    sal_range_str = tier_ranges.get(tier_name, "N/A")
    note          = tier_notes.get(tier_name, "")

    # All band probabilities for frontend chart
    all_probs = {tier_names[i]: round(float(p) * 100, 1)
                 for i, p in enumerate(tier_proba)
                 if i < len(tier_names)}

    result = {
        "predicted_salary": midpoint,
        "salary_band":      tier_name,
        "salary_range":     sal_range_str,
        "confidence":       confidence,
        "percentile_note":  note,
        "all_band_probs":   all_probs,
        "model_type":       salary_meta.get("model_type", "binary_tier"),
    }

    uid = get_jwt_identity()
    db  = Session()
    db.add(Prediction(user_id=uid, type="salary",
                      input_data=str(d), result=str(result)))
    db.commit()
    return jsonify(result), 200


@app.route("/api/resume/score", methods=["POST"])
@jwt_required()
def resume_score():
    """
    Accepts multipart/form-data:
      - file: PDF resume
      - target_role: (optional) 'software_engineer' | 'data_scientist' | ...
    OR JSON:
      - text: plain resume text
      - target_role: (optional)
    """
    target_role = request.form.get("target_role") or request.json.get("target_role") if request.json else None

    if "file" in request.files:
        f = request.files["file"]
        with pdfplumber.open(io.BytesIO(f.read())) as pdf:
            resume_text = " ".join(p.extract_text() or "" for p in pdf.pages)
    elif request.json and "text" in request.json:
        resume_text = request.json["text"]
    else:
        return jsonify({"error": "Provide a PDF file or text field"}), 400

    result = score_resume_text(resume_text, target_role)

    uid = get_jwt_identity()
    db  = Session()
    db.add(Prediction(user_id=uid, type="resume",
                      input_data=f"role:{target_role}", result=str(result)))
    db.commit()

    return jsonify(result), 200

# ─────────────────────────────────────────────────────────────────────────────
#  MOCK INTERVIEW CHATBOT
#  Works fully offline — no API key needed.
#  Uses a built-in question bank + keyword-based feedback engine.
#  If ANTHROPIC_API_KEY or OPENAI_API_KEY is set in .env, uses that instead.
# ─────────────────────────────────────────────────────────────────────────────

# Built-in question bank — used when no API key is available
INTERVIEW_QUESTIONS = {
    "software_engineer": [
        "Tell me about yourself and your interest in software engineering.",
        "Explain a project you built from scratch. What technologies did you use and why?",
        "What is the difference between a stack and a queue? Give a real-world example of each.",
        "How would you find the second largest element in an unsorted array? Walk me through your approach.",
        "Describe a situation where you had to debug a difficult problem. How did you approach it?",
    ],
    "data_scientist": [
        "Tell me about yourself and what drew you to data science.",
        "Explain the difference between overfitting and underfitting. How do you handle each?",
        "Walk me through how you would approach a new machine learning problem end to end.",
        "What is cross-validation and why is it important?",
        "Describe a data analysis or ML project you worked on. What was the outcome?",
    ],
    "marketing_mba": [
        "Tell me about yourself and your interest in marketing.",
        "How would you launch a new product in a competitive market? Walk me through your strategy.",
        "What is the difference between B2B and B2C marketing? Give examples of each.",
        "Describe a situation where you had to persuade someone using data or analysis.",
        "Where do you see marketing heading in the next 5 years?",
    ],
    "finance_mba": [
        "Tell me about yourself and your interest in finance.",
        "Walk me through a DCF valuation. What are its key assumptions and limitations?",
        "What is working capital and why does it matter to a business?",
        "How would you evaluate whether a company is a good investment?",
        "Describe a situation where you analysed financial data to make a decision.",
    ],
}

FEEDBACK_TEMPLATES = [
    "Good answer! You explained your point clearly. {tip}",
    "That's a solid response. {tip}",
    "Nice — you covered the key points. {tip}",
    "Good start. {tip}",
    "Well answered. {tip}",
]

TIPS = {
    "software_engineer": [
        "Try to mention specific technologies or complexity (e.g. time/space) in technical answers.",
        "Use the STAR method (Situation, Task, Action, Result) for behavioural questions.",
        "Quantify your impact when describing projects — e.g. 'reduced load time by 40%'.",
        "Mention teamwork or collaboration where relevant — companies value both.",
        "Back up your answers with a concrete example from your experience.",
    ],
    "data_scientist": [
        "Always mention how you would validate your model, not just build it.",
        "Interviewers love it when you discuss trade-offs between approaches.",
        "Mention real libraries (pandas, sklearn, PyTorch) to show practical experience.",
        "Discuss the business impact of your analysis, not just the technical steps.",
        "Show that you think about data quality and edge cases.",
    ],
    "marketing_mba": [
        "Frame your answers around customer value — what problem does this solve?",
        "Use real brand examples to back up your points.",
        "Show you understand metrics — CAC, LTV, ROI, conversion rates.",
        "Demonstrate that you can back creative ideas with data.",
        "Mention digital channels (SEO, social, email) alongside traditional ones.",
    ],
    "finance_mba": [
        "Always explain the intuition behind a formula, not just the formula itself.",
        "Show you understand the limitations of financial models.",
        "Use real company examples where possible.",
        "Mention risk management alongside return — interviewers value balanced thinking.",
        "Be ready to do back-of-envelope estimates — show quantitative comfort.",
    ],
}

def _builtin_interview(role: str, message: str, history: list) -> str:
    """
    Fully offline interview engine — no API key needed.
    Tracks question number from history, gives contextual feedback,
    and produces a final score after 5 questions.
    """
    import random

    role_key = role.lower().replace(" ", "_").replace("-", "_")
    if role_key not in INTERVIEW_QUESTIONS:
        role_key = "software_engineer"

    questions = INTERVIEW_QUESTIONS[role_key]
    tips      = TIPS[role_key]

    # Count assistant messages to track progress
    assistant_msgs = [m for m in history if m["role"] == "assistant"]
    n_asked = len(assistant_msgs)

    # First message — intro + Q1
    if message.strip().lower() in ("start", ""):
        role_display = role.replace("_", " ").title()
        lines = [
            "Welcome to your mock placement interview for the role of **" + role_display + "**!",
            "",
            "I will ask you 5 questions and give feedback after each one. Take your time and answer naturally.",
            "",
            "**Question 1 of 5:**",
            questions[0]
        ]
        return "\n".join(lines)

    # After all 5 questions answered — final assessment
    if n_asked >= 6:
        score     = random.randint(6, 9)
        strength  = random.choice(["communication clarity", "structured thinking",
                                   "use of examples", "technical depth"])
        improve   = random.choice(["quantifying your impact with numbers",
                                   "mentioning more specific tools and technologies",
                                   "using the STAR method more consistently",
                                   "showing more awareness of industry trends"])
        if score >= 8:
            overall = "Strong performance — you are well prepared!"
        else:
            overall = "Good effort — a bit more practice and you will be very competitive."
        lines = [
            "That completes our mock interview! Here is your overall assessment:",
            "",
            "**Score: " + str(score) + "/10**",
            "",
            "**Strengths:** You showed good " + strength + " throughout the interview.",
            "",
            "**Area to improve:** Focus on " + improve + " in your answers.",
            "",
            "**Overall:** " + overall,
            "",
            "Type 'start' or click Restart to try again."
        ]
        return "\n".join(lines)

    # Give feedback on the user answer, then ask next question
    answer_len = len(message.split())
    tip        = tips[min(n_asked, len(tips) - 1)]

    if answer_len < 10:
        feedback = "Your answer was quite brief. Try to elaborate more — aim for at least 3-4 sentences. " + tip
    elif answer_len < 25:
        feedback = random.choice(FEEDBACK_TEMPLATES[:3]).format(tip=tip)
    else:
        feedback = random.choice(FEEDBACK_TEMPLATES).format(tip=tip)

    next_q_idx = min(n_asked, len(questions) - 1)
    next_q_num = n_asked + 1

    lines = [
        feedback,
        "",
        "**Question " + str(next_q_num) + " of 5:**",
        questions[next_q_idx]
    ]
    return "\n".join(lines)

@app.route("/api/interview/chat", methods=["POST"])
@jwt_required()
def interview_chat():
    """
    Input JSON:
    {
      "role": "software_engineer",
      "message": "User answer or 'start'",
      "history": [ {"role":"assistant","content":"..."}, ... ]
    }
    """
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    CLAUDE_KEY = os.getenv("ANTHROPIC_API_KEY")

    data    = request.json
    role    = data.get("role", "software_engineer")
    message = data.get("message", "start")
    history = data.get("history", [])

    reply = None

    # ── Try Claude API ──────────────────────────────────────────────────────
    if CLAUDE_KEY:
        try:
            system_prompt = (
                f"You are a professional interviewer for the role of {role.replace('_',' ')}. "
                "Ask one question at a time. After each answer give 2-3 sentences of feedback "
                "then ask the next question. After 5 questions give a score out of 10. "
                "Keep it relevant to Indian campus placements."
            )
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": CLAUDE_KEY,
                         "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 512,
                      "system": system_prompt,
                      "messages": history + [{"role":"user","content":message}]},
                timeout=10
            )
            reply = resp.json()["content"][0]["text"]
        except Exception as e:
            print(f"Claude API error: {e} — falling back to built-in engine")

    # ── Try OpenAI API ──────────────────────────────────────────────────────
    if reply is None and OPENAI_KEY:
        try:
            system_prompt = (
                f"You are a professional interviewer for the role of {role.replace('_',' ')}. "
                "Ask one question at a time, give brief feedback, then ask the next. "
                "After 5 questions give a score out of 10."
            )
            msgs = [{"role":"system","content":system_prompt}]
            msgs.extend(history)
            msgs.append({"role":"user","content":message})
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_KEY}",
                         "Content-Type": "application/json"},
                json={"model": "gpt-3.5-turbo", "messages": msgs, "max_tokens": 512},
                timeout=10
            )
            reply = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"OpenAI API error: {e} — falling back to built-in engine")

    # ── Built-in offline engine (always works, no key needed) ───────────────
    if reply is None:
        reply = _builtin_interview(role, message, history)

    return jsonify({"reply": reply}), 200

# ─────────────────────────────────────────────────────────────────────────────
#  HISTORY
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/history", methods=["GET"])
@jwt_required()
def get_history():
    uid = get_jwt_identity()
    db  = Session()
    preds = db.query(Prediction).filter_by(user_id=uid)\
               .order_by(Prediction.timestamp.desc()).limit(20).all()
    return jsonify([{
        "id": p.id, "type": p.type,
        "timestamp": p.timestamp.isoformat(),
        "result": p.result
    } for p in preds]), 200

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": placement_model is not None}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)
