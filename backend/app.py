"""
=============================================================
  FLASK REST API — UPGRADED VERSION
  AI Placement Suite | College Project
=============================================================
New in this version:
  - /api/resume/analyze  — LLM-powered resume analysis (Claude)
  - /api/models/comparison — returns model comparison data
  - /api/predict/salary   — returns full tier details for UI
  - All existing endpoints unchanged
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
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "dev-secret-change-in-prod")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)
jwt = JWTManager(app)

# ── Database ──────────────────────────────────────────────────────────────────
engine = create_engine("sqlite:///placement.db", echo=False)
Base   = declarative_base()
Session= sessionmaker(bind=engine)

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
    type       = Column(String(50))
    input_data = Column(Text)
    result     = Column(Text)
    timestamp  = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# ── Load models ───────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models")

try:
    placement_model    = joblib.load(os.path.join(MODEL_PATH, "placement_model.pkl"))
    placement_feats    = joblib.load(os.path.join(MODEL_PATH, "placement_features.pkl"))
    salary_model       = joblib.load(os.path.join(MODEL_PATH, "salary_model.pkl"))
    salary_feats       = joblib.load(os.path.join(MODEL_PATH, "salary_features.pkl"))
    salary_meta        = joblib.load(os.path.join(MODEL_PATH, "salary_meta.pkl"))
    resume_scorer_pkg  = joblib.load(os.path.join(MODEL_PATH, "resume_scorer.pkl"))
    vectorizer, tfidf_matrix, categories, KEYWORD_BANK = resume_scorer_pkg

    # Load comparison results if available
    _cmp_path = os.path.join(MODEL_PATH, "comparison_results.json")
    comparison_data = json.load(open(_cmp_path)) if os.path.exists(_cmp_path) else []

    # Best model name
    _bmn_path = os.path.join(MODEL_PATH, "best_model_name.pkl")
    best_model_name = joblib.load(_bmn_path) if os.path.exists(_bmn_path) else "Random Forest"

    print(f"✅ All models loaded. Best placement model: {best_model_name}")
    print(f"   Salary model: {salary_meta.get('model_type','unknown')}")
except FileNotFoundError as e:
    print(f"⚠️  Model not found: {e}. Run 02_train_models.py first.")
    placement_model = salary_model = None
    salary_meta = {}
    comparison_data = []
    best_model_name = "Unknown"

# ── Helpers ───────────────────────────────────────────────────────────────────
def _compute_academic_score(d):
    return (0.25*d["ssc_p"] + 0.25*d["hsc_p"] +
            0.30*d["degree_p"] + 0.20*d["etest_p"])

def score_resume_text(resume_text, target_role=None):
    from sklearn.metrics.pairwise import cosine_similarity
    resume_lower = resume_text.lower()
    rv   = vectorizer.transform([resume_lower])
    sims = cosine_similarity(rv, tfidf_matrix)[0]
    role_scores = {cat: round(float(s)*100,1) for cat,s in zip(categories,sims)}
    best_match  = max(role_scores, key=role_scores.get)
    kws         = KEYWORD_BANK.get(target_role or best_match, [])
    present     = [k for k in kws if k in resume_lower]
    missing     = [k for k in kws if k not in resume_lower]
    return {
        "overall_score":    round(len(present)/len(kws)*100,1) if kws else 0,
        "best_match":       best_match,
        "role_scores":      role_scores,
        "present_keywords": present,
        "missing_keywords": missing[:10]
    }

# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.json
    db   = Session()
    if db.query(User).filter_by(email=data["email"]).first():
        return jsonify({"error": "Email already registered"}), 409
    user = User(name=data["name"], email=data["email"],
                password=generate_password_hash(data["password"]))
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

# ── Placement Predictor ───────────────────────────────────────────────────────
@app.route("/api/predict/placement", methods=["POST"])
@jwt_required()
def predict_placement():
    if placement_model is None:
        return jsonify({"error": "Model not trained"}), 503
    d = request.json
    d["academic_score"] = _compute_academic_score(d)
    X    = pd.DataFrame([[d[f] for f in placement_feats]], columns=placement_feats)
    prob = float(placement_model.predict_proba(X)[0][1])
    pred = int(placement_model.predict(X)[0])

    advice = []
    if d["academic_score"] < 65:
        advice.append("Focus on improving your academic performance — academic score is the top predictor.")
    if d["workex"] == 0:
        advice.append("Gain internship or project experience — work experience significantly boosts placement odds.")
    if d["mba_p"] < 65:
        advice.append("Strengthen your MBA coursework — MBA % is a key feature for placement prediction.")
    if d["etest_p"] < 65:
        advice.append("Work on aptitude and entrance test preparation.")
    if not advice:
        advice.append("Excellent profile! You have a strong chance of placement.")

    result = {
        "placed":       pred == 1,
        "probability":  round(prob * 100, 1),
        "confidence":   "High" if abs(prob-0.5) > 0.3 else "Medium" if abs(prob-0.5) > 0.1 else "Low",
        "advice":       advice,
        "model_used":   best_model_name,
    }
    db = Session(); uid = get_jwt_identity()
    db.add(Prediction(user_id=uid, type="placement",
                      input_data=str(d), result=str(result)))
    db.commit()
    return jsonify(result), 200

# ── Salary Tier Predictor ─────────────────────────────────────────────────────
@app.route("/api/predict/salary", methods=["POST"])
@jwt_required()
def predict_salary():
    if salary_model is None:
        return jsonify({"error": "Model not trained"}), 503
    d = request.json
    d["academic_score"] = _compute_academic_score(d)
    X = pd.DataFrame([[d[f] for f in salary_feats]], columns=salary_feats)

    tier_idx   = int(salary_model.predict(X)[0])
    tier_proba = salary_model.predict_proba(X)[0]
    confidence = round(float(tier_proba[tier_idx]) * 100, 1)

    tier_names    = salary_meta.get("tier_names",    ["Low","High"])
    tier_ranges   = salary_meta.get("tier_ranges",   {"Low":"₹2L–₹2.65L","High":"₹2.65L–₹9.4L"})
    tier_midpoints= salary_meta.get("tier_midpoints",{"Low":232500,"High":602500})
    tier_notes    = salary_meta.get("tier_notes",    {"Low":"Below median","High":"Above median"})

    tier_name  = tier_names[tier_idx] if tier_idx < len(tier_names) else "Unknown"
    midpoint   = tier_midpoints.get(tier_name, 265000)
    all_probs  = {tier_names[i]: round(float(p)*100,1)
                  for i,p in enumerate(tier_proba) if i < len(tier_names)}

    # Feature contribution breakdown (which features pushed toward High)
    feature_tips = []
    if d.get("workex", 0) == 1:
        feature_tips.append("Work experience is boosting your salary tier.")
    if d.get("mba_p", 0) >= 70:
        feature_tips.append("Strong MBA % is a positive signal for salary.")
    if d.get("etest_p", 0) >= 75:
        feature_tips.append("High entrance test score correlates with better packages.")
    if not feature_tips:
        feature_tips.append("Improving MBA % and entrance test score most impacts salary tier.")

    result = {
        "predicted_salary": midpoint,
        "salary_band":      tier_name,
        "salary_range":     tier_ranges.get(tier_name, "N/A"),
        "confidence":       confidence,
        "percentile_note":  tier_notes.get(tier_name, ""),
        "all_band_probs":   all_probs,
        "feature_tips":     feature_tips,
        "model_used":       salary_meta.get("best_model","Random Forest"),
        "median_salary":    salary_meta.get("median", 265000),
    }
    db = Session(); uid = get_jwt_identity()
    db.add(Prediction(user_id=uid, type="salary",
                      input_data=str(d), result=str(result)))
    db.commit()
    return jsonify(result), 200

# ── Resume Scorer (TF-IDF) ────────────────────────────────────────────────────
@app.route("/api/resume/score", methods=["POST"])
@jwt_required()
def resume_score():
    target_role = None
    resume_text = ""

    if request.content_type and "multipart" in request.content_type:
        # PDF upload
        target_role = request.form.get("target_role")
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        f = request.files["file"]
        try:
            with pdfplumber.open(io.BytesIO(f.read())) as pdf:
                resume_text = " ".join(p.extract_text() or "" for p in pdf.pages)
        except Exception as e:
            return jsonify({"error": f"PDF read failed: {str(e)}"}), 400
    else:
        data = request.get_json(force=True) or {}
        resume_text = data.get("text", "")
        target_role = data.get("target_role")

    if not resume_text.strip():
        return jsonify({"error": "No text found. Paste text or upload a valid PDF."}), 400

    result = score_resume_text(resume_text, target_role)

    db = Session(); uid = get_jwt_identity()
    db.add(Prediction(user_id=uid, type="resume",
                      input_data=f"role:{target_role}", result=str(result)))
    db.commit()
    return jsonify(result), 200


# ── LLM Resume Analyzer (NEW) ─────────────────────────────────────────────────
@app.route("/api/resume/analyze", methods=["POST"])
@jwt_required()
def resume_analyze():
    """
    AI-powered resume analysis using Claude API.
    Falls back to rule-based analysis if no API key.
    Input: { "text": "...", "target_role": "software_engineer" }
    Or multipart with file upload.
    """
    CLAUDE_KEY = os.getenv("ANTHROPIC_API_KEY")

    target_role = None
    resume_text = ""

    if request.content_type and "multipart" in request.content_type:
        target_role = request.form.get("target_role", "software_engineer")
        if "file" in request.files:
            f = request.files["file"]
            try:
                with pdfplumber.open(io.BytesIO(f.read())) as pdf:
                    resume_text = " ".join(p.extract_text() or "" for p in pdf.pages)
            except Exception as e:
                return jsonify({"error": f"PDF read failed: {str(e)}"}), 400
    else:
        data = request.get_json(force=True) or {}
        resume_text = data.get("text", "")
        target_role = data.get("target_role", "software_engineer")

    if not resume_text.strip():
        return jsonify({"error": "No resume text provided"}), 400

    # TF-IDF score first (always run)
    tfidf_result = score_resume_text(resume_text, target_role)

    # LLM analysis
    llm_feedback = None
    used_llm     = False

    if CLAUDE_KEY:
        try:
            prompt = f"""You are an expert resume reviewer for campus placements in India.
Analyze this resume for the role of {target_role.replace('_',' ')}.

Resume text:
{resume_text[:3000]}

Provide a structured analysis with exactly these sections:
1. OVERALL SCORE: X/10
2. STRENGTHS: (3 bullet points of what is good)
3. WEAKNESSES: (3 bullet points of what needs improvement)
4. MISSING SKILLS: (5 specific skills/keywords missing for this role)
5. IMPROVEMENT TIPS: (3 actionable tips to improve this resume)
6. ATS SCORE: X/10 (how well it would pass Applicant Tracking Systems)

Be specific and practical. Focus on Indian campus placement context."""

            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": CLAUDE_KEY,
                         "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001",
                      "max_tokens": 800,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=15
            )
            if resp.status_code == 200:
                llm_feedback = resp.json()["content"][0]["text"]
                used_llm     = True
        except Exception as e:
            print(f"Claude API error: {e}")

    # Fallback: rule-based feedback
    if not llm_feedback:
        score       = tfidf_result["overall_score"]
        present_n   = len(tfidf_result["present_keywords"])
        missing_top = tfidf_result["missing_keywords"][:5]
        role_clean  = target_role.replace("_", " ").title()
        lines = [
            f"1. OVERALL SCORE: {min(10, max(1, round(score/10)))}/10",
            "",
            "2. STRENGTHS:",
            f"   - Found {present_n} relevant keywords for {role_clean} role",
            f"   - Resume text is readable and parseable",
            f"   - Covers {round(score)}% of key skills for this role",
            "",
            "3. WEAKNESSES:",
            f"   - Missing important keywords: {', '.join(missing_top[:3])}",
            "   - Could benefit from more quantified achievements",
            "   - Add more role-specific technical skills",
            "",
            "4. MISSING SKILLS:",
            *[f"   - {kw}" for kw in missing_top],
            "",
            "5. IMPROVEMENT TIPS:",
            "   - Add the missing keywords naturally into your experience bullets",
            "   - Quantify achievements with numbers (e.g. 'improved performance by 30%')",
            "   - Tailor your resume summary to the specific role",
            "",
            f"6. ATS SCORE: {min(10, max(1, round(score/12)))}/10",
            "",
            "Note: Add your ANTHROPIC_API_KEY to .env for detailed AI-powered analysis."
        ]
        llm_feedback = "\n".join(lines)

    result = {
        **tfidf_result,
        "llm_feedback": llm_feedback,
        "used_llm":     used_llm,
        "analysis_type": "Claude AI Analysis" if used_llm else "Rule-based Analysis",
    }

    db = Session(); uid = get_jwt_identity()
    db.add(Prediction(user_id=uid, type="resume_llm",
                      input_data=f"role:{target_role}", result=str(result)[:500]))
    db.commit()
    return jsonify(result), 200


# ── Model Comparison ──────────────────────────────────────────────────────────
@app.route("/api/models/comparison", methods=["GET"])
@jwt_required()
def get_model_comparison():
    """Returns model comparison data for the frontend chart."""
    return jsonify({
        "placement_comparison": [r for r in comparison_data if "model" in r and "feature_importance" not in r],
        "best_model":           best_model_name,
        "salary_comparison":    salary_meta.get("salary_comparison", []),
    }), 200


# ── Interview Chatbot ─────────────────────────────────────────────────────────
INTERVIEW_QUESTIONS = {
    "software_engineer": [
        "Tell me about yourself and your interest in software engineering.",
        "Explain a project you built from scratch. What technologies did you use and why?",
        "What is the difference between a stack and a queue? Give a real-world example.",
        "How would you find the second largest element in an unsorted array?",
        "Describe a time you debugged a difficult problem. What was your approach?",
    ],
    "data_scientist": [
        "Tell me about yourself and what drew you to data science.",
        "Explain the difference between overfitting and underfitting. How do you handle each?",
        "Walk me through how you would approach a new ML problem end to end.",
        "What is cross-validation and why is it important?",
        "Describe a data analysis project you worked on. What was the outcome?",
    ],
    "marketing_mba": [
        "Tell me about yourself and your interest in marketing.",
        "How would you launch a new product in a competitive market?",
        "What is the difference between B2B and B2C marketing?",
        "Describe a situation where you used data to make a marketing decision.",
        "Where do you see marketing heading in the next 5 years?",
    ],
    "finance_mba": [
        "Tell me about yourself and your interest in finance.",
        "Walk me through a DCF valuation and its key assumptions.",
        "What is working capital and why does it matter?",
        "How would you evaluate whether a company is a good investment?",
        "Describe a situation where you analysed financial data to make a decision.",
    ],
}

FEEDBACK_TEMPLATES = [
    "Good answer! {tip}",
    "That's a solid response. {tip}",
    "Nice — you covered the key points. {tip}",
    "Good start. {tip}",
    "Well answered. {tip}",
]

TIPS = {
    "software_engineer": [
        "Try to mention specific technologies or time/space complexity.",
        "Use the STAR method for behavioural questions.",
        "Quantify your impact — e.g. 'reduced load time by 40%'.",
        "Mention teamwork or collaboration where relevant.",
        "Back up your answers with a concrete example.",
    ],
    "data_scientist": [
        "Always mention how you would validate your model.",
        "Discuss trade-offs between approaches.",
        "Mention real libraries (pandas, sklearn, PyTorch).",
        "Discuss the business impact of your analysis.",
        "Show you think about data quality and edge cases.",
    ],
    "marketing_mba": [
        "Frame answers around customer value.",
        "Use real brand examples to back up your points.",
        "Show you understand metrics — CAC, LTV, ROI.",
        "Demonstrate you can back creative ideas with data.",
        "Mention digital channels alongside traditional ones.",
    ],
    "finance_mba": [
        "Explain the intuition behind formulas, not just the formula.",
        "Show you understand the limitations of financial models.",
        "Use real company examples where possible.",
        "Mention risk management alongside return.",
        "Show quantitative comfort with back-of-envelope estimates.",
    ],
}

def _builtin_interview(role, message, history):
    import random
    role_key   = role.lower().replace(" ", "_").replace("-", "_")
    if role_key not in INTERVIEW_QUESTIONS:
        role_key = "software_engineer"
    questions  = INTERVIEW_QUESTIONS[role_key]
    tips       = TIPS[role_key]
    n_asked    = len([m for m in history if m["role"] == "assistant"])

    if message.strip().lower() in ("start", ""):
        role_display = role.replace("_", " ").title()
        parts = [
            "Welcome to your mock placement interview for the role of **" + role_display + "**!",
            "",
            "I will ask you 5 questions and give feedback after each one. Take your time.",
            "",
            "**Question 1 of 5:**",
            questions[0]
        ]
        return "\n".join(parts)

    if n_asked >= 6:
        score    = random.randint(6, 9)
        strength = random.choice(["communication clarity","structured thinking","use of examples","technical depth"])
        improve  = random.choice(["quantifying impact with numbers","mentioning specific tools","using STAR method consistently","showing industry awareness"])
        overall  = "Strong performance — well prepared!" if score >= 8 else "Good effort — more practice and you will be very competitive."
        parts    = [
            "That completes our mock interview! Here is your overall assessment:",
            "", "**Score: " + str(score) + "/10**", "",
            "**Strengths:** You showed good " + strength + " throughout.",
            "", "**Area to improve:** Focus on " + improve + ".",
            "", "**Overall:** " + overall, "",
            "Type 'start' or click Restart to try again."
        ]
        return "\n".join(parts)

    answer_len = len(message.split())
    tip        = tips[min(n_asked, len(tips)-1)]
    if answer_len < 10:
        feedback = "Your answer was quite brief — try to elaborate more (3-4 sentences minimum). " + tip
    elif answer_len < 25:
        feedback = random.choice(FEEDBACK_TEMPLATES[:3]).format(tip=tip)
    else:
        feedback = random.choice(FEEDBACK_TEMPLATES).format(tip=tip)

    next_q_idx = min(n_asked, len(questions)-1)
    next_q_num = n_asked + 1
    parts = [feedback, "", "**Question " + str(next_q_num) + " of 5:**", questions[next_q_idx]]
    return "\n".join(parts)


@app.route("/api/interview/chat", methods=["POST"])
@jwt_required()
def interview_chat():
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    CLAUDE_KEY = os.getenv("ANTHROPIC_API_KEY")
    data    = request.json
    role    = data.get("role", "software_engineer")
    message = data.get("message", "start")
    history = data.get("history", [])
    reply   = None

    if CLAUDE_KEY:
        try:
            system_prompt = (
                f"You are a professional interviewer for {role.replace('_',' ')} campus placement. "
                "Ask one question at a time. Give 2-3 sentences feedback after each answer, "
                "then ask the next question. After 5 questions give a score out of 10."
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
            if resp.status_code == 200:
                reply = resp.json()["content"][0]["text"]
        except Exception as e:
            print(f"Claude API error: {e}")

    if reply is None and OPENAI_KEY:
        try:
            msgs = [{"role":"system","content":f"Mock interviewer for {role}. One question at a time."}]
            msgs.extend(history)
            msgs.append({"role":"user","content":message})
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization":f"Bearer {OPENAI_KEY}","Content-Type":"application/json"},
                json={"model":"gpt-3.5-turbo","messages":msgs,"max_tokens":512},
                timeout=10
            )
            if resp.status_code == 200:
                reply = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"OpenAI error: {e}")

    if reply is None:
        reply = _builtin_interview(role, message, history)

    return jsonify({"reply": reply}), 200


# ── History ───────────────────────────────────────────────────────────────────
@app.route("/api/history", methods=["GET"])
@jwt_required()
def get_history():
    uid   = get_jwt_identity()
    db    = Session()
    preds = db.query(Prediction).filter_by(user_id=uid)\
               .order_by(Prediction.timestamp.desc()).limit(20).all()
    return jsonify([{
        "id": p.id, "type": p.type,
        "timestamp": p.timestamp.isoformat(),
        "result": p.result
    } for p in preds]), 200

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": placement_model is not None,
        "best_model": best_model_name,
    }), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)