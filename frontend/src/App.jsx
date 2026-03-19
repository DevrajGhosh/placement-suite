/**
 * =============================================================
 *  PHASE 4 — REACT FRONTEND
 *  AI Placement Suite | College Project
 * =============================================================
 *
 *  Setup:
 *    npx create-react-app placement-frontend
 *    cd placement-frontend
 *    npm install axios recharts react-router-dom react-dropzone
 *    Replace src/App.js with this file
 *
 *  Features:
 *    1. Placement Predictor
 *    2. Salary Predictor
 *    3. Resume Analyzer (PDF upload)
 *    4. Mock Interview Chatbot
 *    5. Dashboard with history charts
 * =============================================================
 */

import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell
} from "recharts";

const API = "https://placement-suite-api.onrender.com/api";

// ── Auth helpers ──────────────────────────────────────────────────────────────
const getToken = () => localStorage.getItem("token");
const authHeader = () => ({ Authorization: `Bearer ${getToken()}` });

// ── Color palette ─────────────────────────────────────────────────────────────
const COLORS = {
  blue:   "#378ADD",
  teal:   "#1D9E75",
  coral:  "#D85A30",
  amber:  "#BA7517",
  purple: "#7F77DD",
  green:  "#639922",
  gray:   "#888780",
};

// ═════════════════════════════════════════════════════════════════════════════
//  COMPONENT: Auth (Login / Register)
// ═════════════════════════════════════════════════════════════════════════════
function Auth({ onLogin }) {
  const [isLogin, setIsLogin] = useState(true);
  const [form, setForm] = useState({ name: "", email: "", password: "" });
  const [error, setError] = useState("");

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    try {
      const url = isLogin ? `${API}/auth/login` : `${API}/auth/register`;
      const { data } = await axios.post(url, form);
      localStorage.setItem("token", data.token);
      localStorage.setItem("user", JSON.stringify(data.user));
      onLogin(data.user);
    } catch (err) {
      setError(err.response?.data?.error || "Something went wrong");
    }
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center",
                  justifyContent: "center", background: "#f7f6f2" }}>
      <div style={{ background: "#fff", borderRadius: 16, padding: "2.5rem",
                    boxShadow: "0 4px 24px rgba(0,0,0,0.08)", width: 380 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 6, color: "#1a1a1a" }}>
          AI Placement Suite
        </h1>
        <p style={{ fontSize: 14, color: "#888", marginBottom: 24 }}>
          {isLogin ? "Sign in to your account" : "Create your account"}
        </p>
        {error && <div style={{ background: "#fff0ee", color: COLORS.coral,
                                padding: "10px 14px", borderRadius: 8, fontSize: 13,
                                marginBottom: 16 }}>{error}</div>}
        <form onSubmit={submit}>
          {!isLogin && (
            <Input label="Full Name" value={form.name}
                   onChange={v => setForm({ ...form, name: v })} />
          )}
          <Input label="Email" type="email" value={form.email}
                 onChange={v => setForm({ ...form, email: v })} />
          <Input label="Password" type="password" value={form.password}
                 onChange={v => setForm({ ...form, password: v })} />
          <Btn type="submit" color={COLORS.blue} style={{ width: "100%", marginTop: 8 }}>
            {isLogin ? "Sign In" : "Register"}
          </Btn>
        </form>
        <p style={{ textAlign: "center", marginTop: 16, fontSize: 13, color: "#888" }}>
          {isLogin ? "No account? " : "Already registered? "}
          <button onClick={() => setIsLogin(!isLogin)}
                  style={{ color: COLORS.blue, border: "none", background: "none",
                           cursor: "pointer", fontWeight: 600 }}>
            {isLogin ? "Register" : "Sign In"}
          </button>
        </p>
      </div>
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
//  COMPONENT: Main App Shell
// ═════════════════════════════════════════════════════════════════════════════
function App() {
  const [user, setUser] = useState(() => {
    const u = localStorage.getItem("user");
    return u ? JSON.parse(u) : null;
  });
  const [tab, setTab] = useState("placement");

  if (!user) return <Auth onLogin={u => setUser(u)} />;

  const tabs = [
    { id: "placement", label: "Placement Predictor", icon: "🎯" },
    { id: "salary",    label: "Salary Predictor",    icon: "💰" },
    { id: "resume",    label: "Resume Analyzer",     icon: "📄" },
    { id: "interview", label: "Mock Interview",      icon: "🎤" },
    { id: "history",   label: "My History",          icon: "📊" },
  ];

  return (
    <div style={{ minHeight: "100vh", background: "#f7f6f2", fontFamily: "sans-serif" }}>
      {/* Top Nav */}
      <nav style={{ background: "#fff", borderBottom: "1px solid #e8e6e0",
                    padding: "0 2rem", display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ fontSize: 16, fontWeight: 700, color: "#1a1a1a", marginRight: 24,
                       padding: "1rem 0" }}>
          🤖 AI Placement Suite
        </span>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
                  style={{
                    padding: "1rem 14px",
                    border: "none", background: "none", cursor: "pointer",
                    fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
                    color: tab === t.id ? COLORS.blue : "#888",
                    borderBottom: tab === t.id ? `2px solid ${COLORS.blue}` : "2px solid transparent",
                    transition: "all 0.15s"
                  }}>
            {t.icon} {t.label}
          </button>
        ))}
        <button onClick={() => { localStorage.clear(); setUser(null); }}
                style={{ marginLeft: "auto", fontSize: 13, color: "#888",
                         border: "none", background: "none", cursor: "pointer",
                         padding: "0.5rem 0" }}>
          Sign Out ↗
        </button>
      </nav>

      {/* Main Content */}
      <div style={{ maxWidth: 900, margin: "0 auto", padding: "2rem 1rem" }}>
        {tab === "placement" && <PlacementPredictor />}
        {tab === "salary"    && <SalaryPredictor />}
        {tab === "resume"    && <ResumeAnalyzer />}
        {tab === "interview" && <MockInterview />}
        {tab === "history"   && <History />}
      </div>
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
//  MODULE 1: Placement Predictor
// ═════════════════════════════════════════════════════════════════════════════
function PlacementPredictor() {
  const [form, setForm] = useState({
    gender: 1, ssc_p: 70, hsc_p: 68, degree_p: 65,
    workex: 0, etest_p: 70, specialisation: 0, mba_p: 65
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    setLoading(true);
    try {
      const { data } = await axios.post(`${API}/predict/placement`,
        form, { headers: authHeader() });
      setResult(data);
    } catch { alert("Prediction failed. Check API."); }
    setLoading(false);
  };

  return (
    <Card title="🎯 Placement Predictor" subtitle="Will I get placed?">
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <Select label="Gender" value={form.gender}
                onChange={v => setForm({ ...form, gender: +v })}
                options={[{ v: 1, l: "Male" }, { v: 0, l: "Female" }]} />
        <Select label="Work Experience" value={form.workex}
                onChange={v => setForm({ ...form, workex: +v })}
                options={[{ v: 0, l: "No" }, { v: 1, l: "Yes" }]} />
        <Select label="MBA Specialisation" value={form.specialisation}
                onChange={v => setForm({ ...form, specialisation: +v })}
                options={[{ v: 0, l: "Mkt & HR" }, { v: 1, l: "Mkt & Finance" }]} />
        <RangeInput label="10th % (SSC)" value={form.ssc_p}
                    onChange={v => setForm({ ...form, ssc_p: +v })} />
        <RangeInput label="12th % (HSC)" value={form.hsc_p}
                    onChange={v => setForm({ ...form, hsc_p: +v })} />
        <RangeInput label="Degree %" value={form.degree_p}
                    onChange={v => setForm({ ...form, degree_p: +v })} />
        <RangeInput label="Entrance Test %" value={form.etest_p}
                    onChange={v => setForm({ ...form, etest_p: +v })} />
        <RangeInput label="MBA %" value={form.mba_p}
                    onChange={v => setForm({ ...form, mba_p: +v })} />
      </div>
      <Btn onClick={submit} color={COLORS.blue} style={{ marginTop: 16 }}
           loading={loading}>
        Predict Placement
      </Btn>

      {result && (
        <div style={{ marginTop: 20 }}>
          <div style={{
            background: result.placed ? "#eaf3de" : "#fff0ee",
            border: `1px solid ${result.placed ? COLORS.green : COLORS.coral}`,
            borderRadius: 12, padding: "1.25rem", marginBottom: 16
          }}>
            <div style={{ fontSize: 28, fontWeight: 700,
                          color: result.placed ? COLORS.green : COLORS.coral }}>
              {result.placed ? "✅ Likely to be Placed" : "⚠️ Placement Risk"}
            </div>
            <div style={{ fontSize: 15, color: "#555", marginTop: 4 }}>
              Probability: <strong>{result.probability}%</strong> &nbsp;|&nbsp;
              Confidence: <strong>{result.confidence}</strong>
            </div>
          </div>
          <div>
            <p style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Recommendations:</p>
            {result.advice.map((a, i) => (
              <div key={i} style={{ padding: "8px 12px", background: "#f7f6f2",
                                    borderRadius: 8, marginBottom: 6, fontSize: 13,
                                    borderLeft: `3px solid ${COLORS.blue}` }}>
                {a}
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
//  MODULE 2: Salary Predictor
// ═════════════════════════════════════════════════════════════════════════════
function SalaryPredictor() {
  const [form, setForm] = useState({
    gender: 1, ssc_p: 70, hsc_p: 68, degree_p: 65,
    workex: 1, etest_p: 75, specialisation: 1, mba_p: 70
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    setLoading(true);
    try {
      const { data } = await axios.post(`${API}/predict/salary`,
        form, { headers: authHeader() });
      setResult(data);
    } catch { alert("Prediction failed."); }
    setLoading(false);
  };

  const bandColors = { "Entry": COLORS.gray, "Lower-Mid": COLORS.amber,
                       "Upper-Mid": COLORS.blue, "Top": COLORS.green };
  const chartData = result ? [
    { name: "SSC", score: form.ssc_p },
    { name: "HSC", score: form.hsc_p },
    { name: "Degree", score: form.degree_p },
    { name: "E-Test", score: form.etest_p },
    { name: "MBA", score: form.mba_p },
  ] : [];

  return (
    <Card title="💰 Salary Predictor" subtitle="Estimate your placement package">
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <Select label="Gender" value={form.gender}
                onChange={v => setForm({ ...form, gender: +v })}
                options={[{ v: 1, l: "Male" }, { v: 0, l: "Female" }]} />
        <Select label="Work Experience" value={form.workex}
                onChange={v => setForm({ ...form, workex: +v })}
                options={[{ v: 0, l: "No" }, { v: 1, l: "Yes" }]} />
        <RangeInput label="10th %" value={form.ssc_p}
                    onChange={v => setForm({ ...form, ssc_p: +v })} />
        <RangeInput label="12th %" value={form.hsc_p}
                    onChange={v => setForm({ ...form, hsc_p: +v })} />
        <RangeInput label="Degree %" value={form.degree_p}
                    onChange={v => setForm({ ...form, degree_p: +v })} />
        <RangeInput label="MBA %" value={form.mba_p}
                    onChange={v => setForm({ ...form, mba_p: +v })} />
      </div>
      <Btn onClick={submit} color={COLORS.teal} style={{ marginTop: 16 }}
           loading={loading}>
        Predict Salary
      </Btn>

      {result && (
        <div style={{ marginTop: 20 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12,
                        marginBottom: 16 }}>
            <StatCard label="Salary Band" value={result.salary_band}
                      color={bandColors[result.salary_band] || COLORS.teal} />
            <StatCard label="Est. Midpoint"
                      value={`₹${Math.round(result.predicted_salary).toLocaleString("en-IN")}`}
                      color={COLORS.blue} />
            <StatCard label="Confidence" value={`${result.confidence || 70}%`}
                      color={COLORS.purple} />
          </div>
          <div style={{ background: "#eaf3de", border: "1px solid #97c459",
                        borderRadius: 10, padding: "10px 14px", marginBottom: 14 }}>
            <p style={{ fontSize: 13, color: "#3b6d11", margin: 0 }}>
              <strong>Salary range:</strong> {result.salary_range} &nbsp;|&nbsp; {result.percentile_note}
            </p>
          </div>
          {result.all_band_probs && (
            <div style={{ marginBottom: 16 }}>
              <p style={{ fontSize: 12, color: "#888", marginBottom: 8 }}>Band probabilities:</p>
              {Object.entries(result.all_band_probs).map(([band, pct]) => (
                <div key={band} style={{ display: "flex", alignItems: "center",
                                         gap: 8, marginBottom: 5 }}>
                  <span style={{ fontSize: 12, width: 90, color: "#555" }}>{band}</span>
                  <div style={{ flex: 1, background: "#eee", borderRadius: 3, height: 6 }}>
                    <div style={{ width: `${pct}%`, background: COLORS.teal,
                                   borderRadius: 3, height: "100%" }} />
                  </div>
                  <span style={{ fontSize: 12, color: "#555", width: 36,
                                  textAlign: "right" }}>{pct}%</span>
                </div>
              ))}
            </div>
          )}
          <ResponsiveContainer width="100%" height={190}>
            <RadarChart data={chartData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="name" tick={{ fontSize: 12 }} />
              <Radar dataKey="score" fill={COLORS.teal} fillOpacity={0.4}
                     stroke={COLORS.teal} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
//  MODULE 3: Resume Analyzer
// ═════════════════════════════════════════════════════════════════════════════
function ResumeAnalyzer() {
  const [text, setText] = useState("");
  const [role, setRole] = useState("data_scientist");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileRef = useRef();

  const analyzeText = async () => {
    setLoading(true);
    try {
      const { data } = await axios.post(`${API}/resume/score`,
        { text, target_role: role }, { headers: authHeader() });
      setResult(data);
    } catch { alert("Analysis failed."); }
    setLoading(false);
  };

  const analyzeFile = async (file) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("target_role", role);
    setLoading(true);
    try {
      const { data } = await axios.post(`${API}/resume/score`, fd,
        { headers: { ...authHeader(), "Content-Type": "multipart/form-data" } });
      setResult(data);
    } catch { alert("Analysis failed."); }
    setLoading(false);
  };

  const rolesChartData = result ? Object.entries(result.role_scores).map(
    ([name, score]) => ({ name: name.replace("_", " "), score })
  ) : [];

  return (
    <Card title="📄 Resume Analyzer" subtitle="Score your resume against job roles">
      <div style={{ marginBottom: 12 }}>
        <label style={{ fontSize: 13, color: "#666", display: "block", marginBottom: 6 }}>
          Target Role
        </label>
        <select value={role} onChange={e => setRole(e.target.value)}
                style={{ padding: "8px 12px", borderRadius: 8, border: "1px solid #ddd",
                         fontSize: 13, width: "100%" }}>
          <option value="software_engineer">Software Engineer</option>
          <option value="data_scientist">Data Scientist</option>
          <option value="marketing_mba">Marketing MBA</option>
          <option value="finance_mba">Finance MBA</option>
        </select>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12,
                    marginBottom: 16 }}>
        <div>
          <label style={{ fontSize: 13, color: "#666", display: "block", marginBottom: 6 }}>
            Paste Resume Text
          </label>
          <textarea value={text} onChange={e => setText(e.target.value)}
                    placeholder="Paste your resume content here..."
                    style={{ width: "100%", height: 140, padding: "10px",
                             borderRadius: 8, border: "1px solid #ddd",
                             fontSize: 13, resize: "vertical", boxSizing: "border-box" }} />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <label style={{ fontSize: 13, color: "#666" }}>Or Upload PDF</label>
          <div onClick={() => fileRef.current.click()}
               style={{ border: "2px dashed #ccc", borderRadius: 12, height: 100,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        cursor: "pointer", color: "#888", fontSize: 13,
                        background: "#fafaf8" }}>
            📎 Click to upload PDF
          </div>
          <input type="file" ref={fileRef} accept=".pdf" style={{ display: "none" }}
                 onChange={e => analyzeFile(e.target.files[0])} />
        </div>
      </div>

      <Btn onClick={analyzeText} color={COLORS.coral} loading={loading}>
        Analyze Resume
      </Btn>

      {result && (
        <div style={{ marginTop: 20 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr",
                        gap: 12, marginBottom: 20 }}>
            <StatCard label="Match Score" value={`${result.overall_score}%`}
                      color={result.overall_score > 60 ? COLORS.green : COLORS.amber} />
            <StatCard label="Best Fit Role"
                      value={result.best_match.replace("_", " ")} color={COLORS.purple} />
            <StatCard label="Keywords Found"
                      value={result.present_keywords.length} color={COLORS.teal} />
          </div>

          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={rolesChartData} layout="vertical">
              <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="name" width={100} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => `${v}%`} />
              <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                {rolesChartData.map((_, i) => (
                  <Cell key={i}
                        fill={Object.values(COLORS)[i % Object.values(COLORS).length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          <div style={{ marginTop: 16 }}>
            <p style={{ fontWeight: 600, fontSize: 13, marginBottom: 8, color: "#c0392b" }}>
              Top Missing Keywords (add these to your resume):
            </p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {result.missing_keywords.map((k, i) => (
                <span key={i} style={{ padding: "4px 10px", background: "#fff0ee",
                                       color: COLORS.coral, borderRadius: 20, fontSize: 12,
                                       border: `1px solid ${COLORS.coral}` }}>
                  {k}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
//  MODULE 4: Mock Interview Chatbot
// ═════════════════════════════════════════════════════════════════════════════
function MockInterview() {
  const [role, setRole] = useState("software_engineer");
  const [history, setHistory] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [started, setStarted] = useState(false);
  const chatEndRef = useRef();

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  const send = async (msg) => {
    const userMsg = { role: "user", content: msg };
    const newHistory = [...history, userMsg];
    setHistory(newHistory);
    setInput("");
    setLoading(true);

    try {
      const { data } = await axios.post(`${API}/interview/chat`,
        { role, message: msg, history: newHistory },
        { headers: authHeader() });
      setHistory(prev => [...prev, { role: "assistant", content: data.reply }]);
    } catch { }
    setLoading(false);
  };

  const start = () => { setStarted(true); setHistory([]); send("start"); };

  return (
    <Card title="🎤 Mock Interview" subtitle="AI-powered placement interview practice">
      {!started ? (
        <div>
          <div style={{ marginBottom: 16 }}>
            <label style={{ fontSize: 13, color: "#666", display: "block", marginBottom: 6 }}>
              Target Role
            </label>
            <select value={role} onChange={e => setRole(e.target.value)}
                    style={{ padding: "8px 12px", borderRadius: 8, border: "1px solid #ddd",
                             fontSize: 13, width: "100%" }}>
              <option value="software_engineer">Software Engineer</option>
              <option value="data_scientist">Data Scientist</option>
              <option value="marketing_mba">MBA - Marketing</option>
              <option value="finance_mba">MBA - Finance</option>
            </select>
          </div>
          <Btn onClick={start} color={COLORS.purple}>Start Interview</Btn>
        </div>
      ) : (
        <div>
          <div style={{ height: 380, overflowY: "auto", padding: "12px",
                        background: "#fafaf8", borderRadius: 12, marginBottom: 12 }}>
            {history.map((msg, i) => (
              <div key={i} style={{
                display: "flex",
                justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
                marginBottom: 12
              }}>
                <div style={{
                  maxWidth: "78%", padding: "10px 14px", borderRadius: 12,
                  fontSize: 13, lineHeight: 1.6,
                  background: msg.role === "user" ? COLORS.blue : "#fff",
                  color: msg.role === "user" ? "#fff" : "#333",
                  border: msg.role === "user" ? "none" : "1px solid #e8e6e0",
                  borderBottomRightRadius: msg.role === "user" ? 4 : 12,
                  borderBottomLeftRadius: msg.role === "user" ? 12 : 4,
                }}>
                  {msg.content}
                </div>
              </div>
            ))}
            {loading && (
              <div style={{ color: "#888", fontSize: 13, padding: "4px 0" }}>
                Interviewer is typing...
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <input value={input} onChange={e => setInput(e.target.value)}
                   onKeyDown={e => e.key === "Enter" && input.trim() && send(input.trim())}
                   placeholder="Type your answer..."
                   style={{ flex: 1, padding: "10px 14px", borderRadius: 8,
                            border: "1px solid #ddd", fontSize: 13 }} />
            <Btn onClick={() => input.trim() && send(input.trim())} color={COLORS.purple}>
              Send
            </Btn>
            <Btn onClick={() => { setStarted(false); setHistory([]); }} color={COLORS.gray}>
              Restart
            </Btn>
          </div>
        </div>
      )}
    </Card>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
//  MODULE 5: History
// ═════════════════════════════════════════════════════════════════════════════
function History() {
  const [records, setRecords] = useState([]);
  useEffect(() => {
    axios.get(`${API}/history`, { headers: authHeader() })
         .then(r => setRecords(r.data));
  }, []);

  const typeColors = { placement: COLORS.blue, salary: COLORS.teal,
                       resume: COLORS.coral, interview: COLORS.purple };

  return (
    <Card title="📊 My Activity History" subtitle="Past predictions and analyses">
      {records.length === 0 ? (
        <p style={{ color: "#888", fontSize: 14 }}>No activity yet. Run a prediction first.</p>
      ) : records.map((r, i) => (
        <div key={i} style={{ padding: "12px 16px", background: "#fafaf8", borderRadius: 10,
                               marginBottom: 10, borderLeft: `4px solid ${typeColors[r.type] || COLORS.gray}` }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: typeColors[r.type] || COLORS.gray,
                           textTransform: "capitalize" }}>
              {r.type}
            </span>
            <span style={{ fontSize: 11, color: "#aaa" }}>
              {new Date(r.timestamp).toLocaleString("en-IN")}
            </span>
          </div>
          <div style={{ fontSize: 12, color: "#888", marginTop: 4 }}>
            {r.result.slice(0, 120)}...
          </div>
        </div>
      ))}
    </Card>
  );
}

// ─── Shared UI components ─────────────────────────────────────────────────────

function Card({ title, subtitle, children }) {
  return (
    <div style={{ background: "#fff", borderRadius: 16, padding: "1.75rem",
                  border: "1px solid #e8e6e0", marginBottom: 24 }}>
      <h2 style={{ fontSize: 18, fontWeight: 700, margin: 0, color: "#1a1a1a" }}>{title}</h2>
      <p style={{ fontSize: 13, color: "#888", marginTop: 4, marginBottom: 20 }}>{subtitle}</p>
      {children}
    </div>
  );
}

function Input({ label, type = "text", value, onChange }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <label style={{ fontSize: 13, color: "#666", display: "block", marginBottom: 4 }}>
        {label}
      </label>
      <input type={type} value={value} onChange={e => onChange(e.target.value)}
             style={{ width: "100%", padding: "9px 12px", borderRadius: 8,
                      border: "1px solid #ddd", fontSize: 14, boxSizing: "border-box" }} />
    </div>
  );
}

function Select({ label, value, onChange, options }) {
  return (
    <div>
      <label style={{ fontSize: 12, color: "#666", display: "block", marginBottom: 4 }}>
        {label}
      </label>
      <select value={value} onChange={e => onChange(e.target.value)}
              style={{ width: "100%", padding: "8px 10px", borderRadius: 8,
                       border: "1px solid #ddd", fontSize: 13 }}>
        {options.map(o => <option key={o.v} value={o.v}>{o.l}</option>)}
      </select>
    </div>
  );
}

function RangeInput({ label, value, onChange }) {
  return (
    <div>
      <label style={{ fontSize: 12, color: "#666", display: "block", marginBottom: 4 }}>
        {label}: <strong>{value}%</strong>
      </label>
      <input type="range" min={40} max={100} step={0.5} value={value}
             onChange={e => onChange(e.target.value)}
             style={{ width: "100%" }} />
    </div>
  );
}

function Btn({ children, onClick, color = COLORS.blue, style = {}, loading, type }) {
  return (
    <button onClick={onClick} type={type || "button"} disabled={loading}
            style={{ padding: "10px 20px", background: color, color: "#fff",
                     border: "none", borderRadius: 8, fontSize: 13, fontWeight: 600,
                     cursor: loading ? "not-allowed" : "pointer",
                     opacity: loading ? 0.7 : 1, transition: "opacity 0.15s", ...style }}>
      {loading ? "..." : children}
    </button>
  );
}

function StatCard({ label, value, color }) {
  return (
    <div style={{ background: "#fafaf8", borderRadius: 10, padding: "12px 14px",
                  borderTop: `3px solid ${color}` }}>
      <div style={{ fontSize: 11, color: "#888", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 17, fontWeight: 700, color }}>{value}</div>
    </div>
  );
}

export default App;
