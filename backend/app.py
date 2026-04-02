from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
import os

app = Flask(__name__)

# ✅ CORS FIX (IMPORTANT)
CORS(
    app,
    resources={r"/*": {"origins": "https://placement-suite-swart.vercel.app"}},
    supports_credentials=True
)

# ✅ ADD THIS (CRITICAL FOR PREFLIGHT)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'https://placement-suite-swart.vercel.app')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    return response


app.config["JWT_SECRET_KEY"] = "dev-secret"
jwt = JWTManager(app)

# ── DB ─────────────────────────────
engine = create_engine("sqlite:///placement.db")
Base = declarative_base()
Session = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    email = Column(String(120), unique=True)
    password = Column(String(256))

Base.metadata.create_all(engine)

# ── AUTH ROUTES (FIXED) ─────────────────────────────

@app.route("/api/auth/register", methods=["POST", "OPTIONS"])
def register():
    # ✅ Handle preflight
    if request.method == "OPTIONS":
        return '', 200

    data = request.json
    db = Session()

    if db.query(User).filter_by(email=data["email"]).first():
        return jsonify({"error": "Email already registered"}), 409

    user = User(
        name=data["name"],
        email=data["email"],
        password=generate_password_hash(data["password"])
    )

    db.add(user)
    db.commit()

    token = create_access_token(identity=str(user.id))

    return jsonify({
        "token": token,
        "user": {"id": user.id, "name": user.name}
    }), 201


@app.route("/api/auth/login", methods=["POST", "OPTIONS"])
def login():
    # ✅ Handle preflight
    if request.method == "OPTIONS":
        return '', 200

    data = request.json
    db = Session()

    user = db.query(User).filter_by(email=data["email"]).first()

    if not user or not check_password_hash(user.password, data["password"]):
        return jsonify({"error": "Invalid credentials"}), 401

    token = create_access_token(identity=str(user.id))

    return jsonify({
        "token": token,
        "user": {"id": user.id, "name": user.name}
    }), 200


# ── TEST ROUTE ─────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(debug=True)