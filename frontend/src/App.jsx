import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, Cell
} from "recharts";

const API = "https://placement-suite-api.onrender.com";
const getToken = () => localStorage.getItem("token");
const authHeader = () => ({ Authorization: `Bearer ${getToken()}` });

const C = {
  blue:"#378ADD", teal:"#1D9E75", coral:"#D85A30",
  amber:"#BA7517", purple:"#7F77DD", green:"#639922", gray:"#888780",
};

// ── Auth ──────────────────────────────────────────────────────────────────────
function Auth({ onLogin }) {
  const [isLogin, setIsLogin] = useState(true);
  const [form, setForm]       = useState({ name:"", email:"", password:"" });
  const [error, setError]     = useState("");

  const submit = async (e) => {
    e.preventDefault(); setError("");
    try {
      const url = isLogin ? `${API}/api/auth/login` : `${API}/api/auth/register`;
      const { data } = await axios.post(url, form);
      localStorage.setItem("token", data.token);
      localStorage.setItem("user", JSON.stringify(data.user));
      onLogin(data.user);
    } catch (err) { setError(err.response?.data?.error || "Something went wrong"); }
  };

  return (
    <div style={{ minHeight:"100vh", display:"flex", alignItems:"center",
                  justifyContent:"center", background:"#f7f6f2" }}>
      <div style={{ background:"#fff", borderRadius:16, padding:"2.5rem",
                    boxShadow:"0 4px 24px rgba(0,0,0,0.08)", width:380 }}>
        <h1 style={{ fontSize:22, fontWeight:700, marginBottom:4 }}>AI Placement Suite</h1>
        <p style={{ fontSize:13, color:"#888", marginBottom:20 }}>
          {isLogin ? "Sign in to your account" : "Create your account"}
        </p>
        {error && <div style={{ background:"#fff0ee", color:C.coral, padding:"10px 14px",
                                borderRadius:8, fontSize:13, marginBottom:14 }}>{error}</div>}
        <form onSubmit={submit}>
          {!isLogin && <Input label="Full Name" value={form.name} onChange={v=>setForm({...form,name:v})}/>}
          <Input label="Email" type="email" value={form.email} onChange={v=>setForm({...form,email:v})}/>
          <Input label="Password" type="password" value={form.password} onChange={v=>setForm({...form,password:v})}/>
          <Btn type="submit" color={C.blue} style={{ width:"100%", marginTop:8 }}>
            {isLogin ? "Sign In" : "Register"}
          </Btn>
        </form>
        <p style={{ textAlign:"center", marginTop:14, fontSize:13, color:"#888" }}>
          {isLogin ? "No account? " : "Already registered? "}
          <button onClick={()=>setIsLogin(!isLogin)} style={{ color:C.blue, border:"none",
            background:"none", cursor:"pointer", fontWeight:600 }}>
            {isLogin ? "Register" : "Sign In"}
          </button>
        </p>
      </div>
    </div>
  );
}

// ── App Shell ─────────────────────────────────────────────────────────────────
function App() {
  const [user, setUser] = useState(()=>{ const u=localStorage.getItem("user"); return u?JSON.parse(u):null; });
  const [tab, setTab]   = useState("placement");
  if (!user) return <Auth onLogin={u=>setUser(u)} />;

  const tabs = [
    { id:"placement",   label:"Placement Predictor", icon:"🎯" },
    { id:"salary",      label:"Salary Predictor",    icon:"💰" },
    { id:"resume",      label:"Resume Analyzer",     icon:"📄" },
    { id:"interview",   label:"Mock Interview",      icon:"🎤" },
    { id:"comparison",  label:"Model Comparison",    icon:"📊" },
  ];

  return (
    <div style={{ minHeight:"100vh", background:"#f7f6f2", fontFamily:"sans-serif" }}>
      <nav style={{ background:"#fff", borderBottom:"1px solid #e8e6e0",
                    padding:"0 1.5rem", display:"flex", alignItems:"center", gap:4,
                    overflowX:"auto" }}>
        <span style={{ fontSize:15, fontWeight:700, marginRight:20, padding:"1rem 0",
                       whiteSpace:"nowrap" }}>🤖 AI Placement Suite</span>
        {tabs.map(t=>(
          <button key={t.id} onClick={()=>setTab(t.id)} style={{
            padding:"1rem 12px", border:"none", background:"none", cursor:"pointer",
            fontSize:12, fontWeight:tab===t.id?600:400, whiteSpace:"nowrap",
            color:tab===t.id?C.blue:"#888",
            borderBottom:tab===t.id?`2px solid ${C.blue}`:"2px solid transparent",
          }}>{t.icon} {t.label}</button>
        ))}
        <button onClick={()=>{ localStorage.clear(); setUser(null); }}
                style={{ marginLeft:"auto", fontSize:12, color:"#888", border:"none",
                         background:"none", cursor:"pointer", whiteSpace:"nowrap" }}>
          Sign Out
        </button>
      </nav>
      <div style={{ maxWidth:960, margin:"0 auto", padding:"1.5rem 1rem" }}>
        {tab==="placement"  && <PlacementPredictor />}
        {tab==="salary"     && <SalaryPredictor />}
        {tab==="resume"     && <ResumeAnalyzer />}
        {tab==="interview"  && <MockInterview />}
        {tab==="comparison" && <ModelComparison />}
      </div>
    </div>
  );
}

// ── Placement Predictor ───────────────────────────────────────────────────────
function PlacementPredictor() {
  const [form, setForm]     = useState({ gender:1,ssc_p:70,hsc_p:68,degree_p:65,
                                          workex:0,etest_p:70,specialisation:0,mba_p:65 });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    setLoading(true);
    try {
      const { data } = await axios.post(`${API}/predict/placement`, form, { headers:authHeader() });
      setResult(data);
    } catch { alert("Prediction failed. Is the backend running?"); }
    setLoading(false);
  };

  return (
    <Card title="🎯 Placement Predictor" subtitle="Predict your campus placement probability">
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12, marginBottom:16 }}>
        <Select label="Gender" value={form.gender} onChange={v=>setForm({...form,gender:+v})}
                options={[{v:1,l:"Male"},{v:0,l:"Female"}]}/>
        <Select label="Work Experience" value={form.workex} onChange={v=>setForm({...form,workex:+v})}
                options={[{v:0,l:"No"},{v:1,l:"Yes"}]}/>
        <Select label="MBA Specialisation" value={form.specialisation} onChange={v=>setForm({...form,specialisation:+v})}
                options={[{v:0,l:"Mkt & HR"},{v:1,l:"Mkt & Finance"}]}/>
        <RangeInput label="10th % (SSC)" value={form.ssc_p} onChange={v=>setForm({...form,ssc_p:+v})}/>
        <RangeInput label="12th % (HSC)" value={form.hsc_p} onChange={v=>setForm({...form,hsc_p:+v})}/>
        <RangeInput label="Degree %" value={form.degree_p} onChange={v=>setForm({...form,degree_p:+v})}/>
        <RangeInput label="Entrance Test %" value={form.etest_p} onChange={v=>setForm({...form,etest_p:+v})}/>
        <RangeInput label="MBA %" value={form.mba_p} onChange={v=>setForm({...form,mba_p:+v})}/>
      </div>
      <Btn onClick={submit} color={C.blue} loading={loading}>Predict Placement</Btn>

      {result && (
        <div style={{ marginTop:20 }}>
          <div style={{ background:result.placed?"#eaf3de":"#fff0ee",
                        border:`1px solid ${result.placed?C.green:C.coral}`,
                        borderRadius:12, padding:"1.25rem", marginBottom:14 }}>
            <div style={{ fontSize:26, fontWeight:700, color:result.placed?C.green:C.coral }}>
              {result.placed ? "✅ Likely to be Placed" : "⚠️ Placement at Risk"}
            </div>
            <div style={{ fontSize:14, color:"#555", marginTop:6 }}>
              Probability: <strong>{result.probability}%</strong> &nbsp;|&nbsp;
              Confidence: <strong>{result.confidence}</strong> &nbsp;|&nbsp;
              Model: <strong>{result.model_used}</strong>
            </div>
            <div style={{ marginTop:12, background:"#fff", borderRadius:8, height:12,
                          overflow:"hidden", border:"1px solid #ddd" }}>
              <div style={{ width:`${result.probability}%`, height:"100%",
                            background:result.placed?C.green:C.coral, transition:"width .5s" }}/>
            </div>
          </div>
          <div>
            <p style={{ fontWeight:600, fontSize:13, marginBottom:8 }}>Recommendations:</p>
            {result.advice.map((a,i)=>(
              <div key={i} style={{ padding:"8px 12px", background:"#f7f6f2", borderRadius:8,
                                    marginBottom:6, fontSize:13, borderLeft:`3px solid ${C.blue}` }}>
                {a}
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

// ── Salary Predictor ──────────────────────────────────────────────────────────
function SalaryPredictor() {
  const [form, setForm]     = useState({ gender:1,ssc_p:70,hsc_p:68,degree_p:65,
                                          workex:1,etest_p:75,specialisation:1,mba_p:70 });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    setLoading(true);
    try {
      const { data } = await axios.post(`${API}/predict/salary`, form, { headers:authHeader() });
      setResult(data);
    } catch { alert("Prediction failed."); }
    setLoading(false);
  };

  const tierColor = { Low:C.amber, High:C.teal };
  const radarData = result ? [
    { name:"SSC",    score:form.ssc_p },
    { name:"HSC",    score:form.hsc_p },
    { name:"Degree", score:form.degree_p },
    { name:"E-Test", score:form.etest_p },
    { name:"MBA",    score:form.mba_p },
  ] : [];

  return (
    <Card title="💰 Salary Tier Predictor" subtitle="Predict your expected salary tier at placement">
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12, marginBottom:16 }}>
        <Select label="Gender" value={form.gender} onChange={v=>setForm({...form,gender:+v})}
                options={[{v:1,l:"Male"},{v:0,l:"Female"}]}/>
        <Select label="Work Experience" value={form.workex} onChange={v=>setForm({...form,workex:+v})}
                options={[{v:0,l:"No"},{v:1,l:"Yes"}]}/>
        <Select label="MBA Specialisation" value={form.specialisation} onChange={v=>setForm({...form,specialisation:+v})}
                options={[{v:0,l:"Mkt & HR"},{v:1,l:"Mkt & Finance"}]}/>
        <RangeInput label="10th % (SSC)" value={form.ssc_p} onChange={v=>setForm({...form,ssc_p:+v})}/>
        <RangeInput label="12th % (HSC)" value={form.hsc_p} onChange={v=>setForm({...form,hsc_p:+v})}/>
        <RangeInput label="Degree %" value={form.degree_p} onChange={v=>setForm({...form,degree_p:+v})}/>
        <RangeInput label="Entrance Test %" value={form.etest_p} onChange={v=>setForm({...form,etest_p:+v})}/>
        <RangeInput label="MBA %" value={form.mba_p} onChange={v=>setForm({...form,mba_p:+v})}/>
      </div>
      <Btn onClick={submit} color={C.teal} loading={loading}>Predict Salary Tier</Btn>

      {result && (
        <div style={{ marginTop:20 }}>
          {/* Main result banner */}
          <div style={{ background:result.salary_band==="High"?"#e1f5ee":"#faeeda",
                        border:`1px solid ${tierColor[result.salary_band]||C.teal}`,
                        borderRadius:12, padding:"1.25rem", marginBottom:14 }}>
            <div style={{ fontSize:24, fontWeight:700,
                          color:tierColor[result.salary_band]||C.teal }}>
              {result.salary_band === "High" ? "📈 High Salary Tier" : "📊 Standard Salary Tier"}
            </div>
            <div style={{ fontSize:14, color:"#555", marginTop:6, lineHeight:1.6 }}>
              <strong>Salary Range:</strong> {result.salary_range} &nbsp;|&nbsp;
              <strong>Est. Midpoint:</strong> ₹{Math.round(result.predicted_salary).toLocaleString("en-IN")} &nbsp;|&nbsp;
              <strong>Confidence:</strong> {result.confidence}%
            </div>
            <div style={{ fontSize:13, color:"#666", marginTop:6 }}>
              {result.percentile_note}
            </div>
          </div>

          {/* Stat cards */}
          <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:10, marginBottom:16 }}>
            <StatCard label="Predicted Tier"
                      value={result.salary_band}
                      color={tierColor[result.salary_band]||C.teal}/>
            <StatCard label="Salary Range"
                      value={result.salary_range}
                      color={C.blue}/>
            <StatCard label="Confidence"
                      value={`${result.confidence}%`}
                      color={C.purple}/>
          </div>

          {/* Band probability bars */}
          {result.all_band_probs && (
            <div style={{ background:"#f7f6f2", borderRadius:10, padding:"1rem",
                          marginBottom:14 }}>
              <p style={{ fontSize:12, fontWeight:600, color:"#666", marginBottom:10 }}>
                Tier probabilities:
              </p>
              {Object.entries(result.all_band_probs).map(([tier,pct])=>(
                <div key={tier} style={{ display:"flex", alignItems:"center",
                                         gap:10, marginBottom:8 }}>
                  <span style={{ fontSize:12, width:48, color:"#555",
                                  fontWeight:tier===result.salary_band?600:400 }}>{tier}</span>
                  <div style={{ flex:1, background:"#ddd", borderRadius:3, height:8 }}>
                    <div style={{ width:`${pct}%`, height:"100%", borderRadius:3,
                                   background:tierColor[tier]||C.gray,
                                   transition:"width .5s" }}/>
                  </div>
                  <span style={{ fontSize:12, color:"#555", width:38,
                                  textAlign:"right" }}>{pct}%</span>
                </div>
              ))}
            </div>
          )}

          {/* Feature tips */}
          {result.feature_tips && result.feature_tips.length > 0 && (
            <div>
              <p style={{ fontSize:12, fontWeight:600, color:"#666", marginBottom:6 }}>
                Key factors:
              </p>
              {result.feature_tips.map((tip,i)=>(
                <div key={i} style={{ padding:"7px 12px", background:"#f0f8f4",
                                      borderRadius:8, marginBottom:5, fontSize:12,
                                      borderLeft:`3px solid ${C.teal}` }}>{tip}</div>
              ))}
            </div>
          )}

          {/* Academic radar */}
          <div style={{ marginTop:16 }}>
            <p style={{ fontSize:12, color:"#888", marginBottom:4 }}>Your academic profile:</p>
            <ResponsiveContainer width="100%" height={180}>
              <RadarChart data={radarData}>
                <PolarGrid/>
                <PolarAngleAxis dataKey="name" tick={{ fontSize:11 }}/>
                <Radar dataKey="score" fill={C.teal} fillOpacity={0.35} stroke={C.teal}/>
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </Card>
  );
}

// ── Resume Analyzer ───────────────────────────────────────────────────────────
function ResumeAnalyzer() {
  const [text, setText]         = useState("");
  const [role, setRole]         = useState("software_engineer");
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [useAI, setUseAI]       = useState(true);
  const [fileName, setFileName] = useState("");
  const fileRef = useRef();

  const analyzeText = async () => {
    if (!text.trim()) { alert("Please paste your resume text first."); return; }
    setLoading(true);
    try {
      const endpoint = useAI ? `${API}/resume/analyze` : `${API}/resume/score`;
      const { data } = await axios.post(endpoint,
        { text, target_role: role }, { headers: authHeader() });
      setResult(data);
    } catch(e) { alert("Analysis failed: " + (e.response?.data?.error || e.message)); }
    setLoading(false);
  };

  const analyzeFile = async (file) => {
    setFileName(file.name);
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("target_role", role);
      const endpoint = useAI ? `${API}/resume/analyze` : `${API}/resume/score`;
      const { data } = await axios.post(endpoint, fd, {
        headers: { ...authHeader() }
      });
      setResult(data);
    } catch(e) { alert("Upload failed: " + (e.response?.data?.error || e.message)); }
    setLoading(false);
  };

  const rolesChart = result ? Object.entries(result.role_scores||{}).map(
    ([name,score])=>({ name:name.replace(/_/g," "), score })
  ) : [];

  return (
    <Card title="📄 Resume Analyzer" subtitle="Score your resume with TF-IDF + optional AI analysis">
      <div style={{ display:"flex", gap:12, marginBottom:14, alignItems:"center" }}>
        <div style={{ flex:1 }}>
          <label style={{ fontSize:12, color:"#666", display:"block", marginBottom:4 }}>Target role</label>
          <select value={role} onChange={e=>setRole(e.target.value)}
                  style={{ width:"100%", padding:"8px 10px", borderRadius:8,
                           border:"1px solid #ddd", fontSize:13 }}>
            <option value="software_engineer">Software Engineer</option>
            <option value="data_scientist">Data Scientist</option>
            <option value="marketing_mba">Marketing MBA</option>
            <option value="finance_mba">Finance MBA</option>
          </select>
        </div>
        <div style={{ paddingTop:18 }}>
          <label style={{ display:"flex", alignItems:"center", gap:6, fontSize:12,
                           cursor:"pointer", color:"#555" }}>
            <input type="checkbox" checked={useAI} onChange={e=>setUseAI(e.target.checked)}/>
            Use AI Analysis (Claude)
          </label>
        </div>
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12, marginBottom:14 }}>
        <div>
          <label style={{ fontSize:12, color:"#666", display:"block", marginBottom:4 }}>
            Paste resume text
          </label>
          <textarea value={text} onChange={e=>setText(e.target.value)}
                    placeholder="Paste your resume content here..."
                    style={{ width:"100%", height:140, padding:"10px", borderRadius:8,
                             border:"1px solid #ddd", fontSize:12, resize:"vertical",
                             boxSizing:"border-box" }}/>
        </div>
        <div>
          <label style={{ fontSize:12, color:"#666", display:"block", marginBottom:4 }}>
            Or upload PDF
          </label>
          <div onClick={()=>fileRef.current.click()}
               style={{ border:"2px dashed #ccc", borderRadius:12, height:112,
                        display:"flex", flexDirection:"column", alignItems:"center",
                        justifyContent:"center", cursor:"pointer", color:"#888",
                        fontSize:12, background:"#fafaf8", gap:6 }}>
            <span style={{ fontSize:24 }}>📎</span>
            {fileName ? <span style={{ color:C.teal, fontWeight:600 }}>{fileName}</span>
                      : <span>Click to upload PDF</span>}
          </div>
          <input type="file" ref={fileRef} accept=".pdf" style={{ display:"none" }}
                 onChange={e=>e.target.files[0] && analyzeFile(e.target.files[0])}/>
        </div>
      </div>

      <Btn onClick={analyzeText} color={C.coral} loading={loading}>
        {useAI ? "Analyze with AI" : "Score Resume"}
      </Btn>

      {result && (
        <div style={{ marginTop:20 }}>
          {/* Stats row */}
          <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:10, marginBottom:16 }}>
            <StatCard label="Match Score" value={`${result.overall_score}%`}
                      color={result.overall_score>60?C.green:C.amber}/>
            <StatCard label="Best Fit Role"
                      value={(result.best_match||"").replace(/_/g," ")} color={C.purple}/>
            <StatCard label="Analysis Type"
                      value={result.analysis_type||"TF-IDF"} color={C.blue}/>
          </div>

          {/* Role scores chart */}
          <div style={{ marginBottom:16 }}>
            <p style={{ fontSize:12, color:"#888", marginBottom:6 }}>Role match scores:</p>
            <ResponsiveContainer width="100%" height={140}>
              <BarChart data={rolesChart} layout="vertical">
                <XAxis type="number" domain={[0,100]} tick={{ fontSize:10 }}/>
                <YAxis type="category" dataKey="name" width={110} tick={{ fontSize:11 }}/>
                <Tooltip formatter={v=>`${v}%`}/>
                <Bar dataKey="score" radius={[0,4,4,0]}>
                  {rolesChart.map((_,i)=>(
                    <Cell key={i} fill={Object.values(C)[i%Object.values(C).length]}/>
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Missing keywords */}
          {result.missing_keywords?.length > 0 && (
            <div style={{ marginBottom:14 }}>
              <p style={{ fontSize:12, fontWeight:600, color:"#c0392b", marginBottom:6 }}>
                Top missing keywords — add these to your resume:
              </p>
              <div style={{ display:"flex", flexWrap:"wrap", gap:6 }}>
                {result.missing_keywords.map((k,i)=>(
                  <span key={i} style={{ padding:"3px 10px", background:"#fff0ee",
                                         color:C.coral, borderRadius:20, fontSize:11,
                                         border:`1px solid ${C.coral}` }}>{k}</span>
                ))}
              </div>
            </div>
          )}

          {/* LLM Feedback */}
          {result.llm_feedback && (
            <div style={{ background:"#f0f4ff", borderRadius:12, padding:"1rem",
                          border:`1px solid ${C.blue}` }}>
              <p style={{ fontSize:12, fontWeight:600, color:C.blue, marginBottom:8 }}>
                {result.used_llm ? "🤖 Claude AI Feedback:" : "📋 Analysis Report:"}
              </p>
              <pre style={{ fontSize:12, color:"#333", whiteSpace:"pre-wrap",
                            fontFamily:"sans-serif", lineHeight:1.7, margin:0 }}>
                {result.llm_feedback}
              </pre>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

// ── Mock Interview ─────────────────────────────────────────────────────────────
function MockInterview() {
  const [role, setRole]       = useState("software_engineer");
  const [history, setHistory] = useState([]);
  const [input, setInput]     = useState("");
  const [loading, setLoading] = useState(false);
  const [started, setStarted] = useState(false);
  const chatEndRef = useRef();

  useEffect(()=>{ chatEndRef.current?.scrollIntoView({ behavior:"smooth" }); }, [history]);

  const send = async (msg) => {
    const userMsg    = { role:"user", content:msg };
    const newHistory = [...history, userMsg];
    setHistory(newHistory); setInput(""); setLoading(true);
    try {
      const { data } = await axios.post(`${API}/interview/chat`,
        { role, message:msg, history:newHistory }, { headers:authHeader() });
      setHistory(prev=>[...prev, { role:"assistant", content:data.reply }]);
    } catch { setHistory(prev=>[...prev, { role:"assistant", content:"Sorry, something went wrong. Please try again." }]); }
    setLoading(false);
  };

  const start = () => { setStarted(true); setHistory([]); send("start"); };

  const formatMsg = (text) => text.split("\n").map((line,i)=>{
    const bold = line.replace(/\*\*(.*?)\*\*/g,"<strong>$1</strong>");
    return <div key={i} dangerouslySetInnerHTML={{ __html:bold||"&nbsp;" }}/>;
  });

  return (
    <Card title="🎤 Mock Interview" subtitle="AI-powered placement interview practice (5 questions + feedback)">
      {!started ? (
        <div>
          <div style={{ marginBottom:14 }}>
            <label style={{ fontSize:12, color:"#666", display:"block", marginBottom:4 }}>Target role</label>
            <select value={role} onChange={e=>setRole(e.target.value)}
                    style={{ width:"100%", padding:"8px 10px", borderRadius:8,
                             border:"1px solid #ddd", fontSize:13 }}>
              <option value="software_engineer">Software Engineer</option>
              <option value="data_scientist">Data Scientist</option>
              <option value="marketing_mba">MBA — Marketing</option>
              <option value="finance_mba">MBA — Finance</option>
            </select>
          </div>
          <div style={{ background:"#f0f4ff", borderRadius:10, padding:"12px 14px",
                        marginBottom:16, fontSize:12, color:"#555", lineHeight:1.6 }}>
            You will be asked 5 role-specific questions. Answer naturally and receive
            feedback after each. A final score out of 10 is given at the end.
          </div>
          <Btn onClick={start} color={C.purple}>Start Interview</Btn>
        </div>
      ) : (
        <div>
          <div style={{ height:380, overflowY:"auto", padding:12, background:"#fafaf8",
                        borderRadius:12, marginBottom:12 }}>
            {history.map((msg,i)=>(
              <div key={i} style={{ display:"flex",
                                    justifyContent:msg.role==="user"?"flex-end":"flex-start",
                                    marginBottom:12 }}>
                <div style={{ maxWidth:"80%", padding:"10px 14px", borderRadius:12,
                              fontSize:13, lineHeight:1.6,
                              background:msg.role==="user"?C.blue:"#fff",
                              color:msg.role==="user"?"#fff":"#333",
                              border:msg.role==="user"?"none":"1px solid #e8e6e0",
                              borderBottomRightRadius:msg.role==="user"?4:12,
                              borderBottomLeftRadius:msg.role==="user"?12:4 }}>
                  {formatMsg(msg.content)}
                </div>
              </div>
            ))}
            {loading && <div style={{ color:"#888", fontSize:12, padding:"4px 0" }}>Interviewer is typing...</div>}
            <div ref={chatEndRef}/>
          </div>
          <div style={{ display:"flex", gap:8 }}>
            <input value={input} onChange={e=>setInput(e.target.value)}
                   onKeyDown={e=>e.key==="Enter"&&input.trim()&&send(input.trim())}
                   placeholder="Type your answer and press Enter..."
                   style={{ flex:1, padding:"10px 14px", borderRadius:8,
                            border:"1px solid #ddd", fontSize:13 }}/>
            <Btn onClick={()=>input.trim()&&send(input.trim())} color={C.purple}>Send</Btn>
            <Btn onClick={()=>{ setStarted(false); setHistory([]); }} color={C.gray}>Restart</Btn>
          </div>
        </div>
      )}
    </Card>
  );
}

// ── Model Comparison ──────────────────────────────────────────────────────────
function ModelComparison() {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(()=>{
    axios.get(`${API}/models/comparison`, { headers:authHeader() })
      .then(r=>setData(r.data))
      .catch(()=>setData(null))
      .finally(()=>setLoading(false));
  }, []);

  if (loading) return <Card title="📊 Model Comparison"><p style={{ color:"#888" }}>Loading...</p></Card>;
  if (!data)   return <Card title="📊 Model Comparison"><p style={{ color:"#888" }}>Run 02_train_models.py first to generate comparison data.</p></Card>;

  const placementData = (data.placement_comparison||[]).map(r=>({
    name: r.model?.replace(" ","\\n")||r.model,
    "CV Accuracy": r.cv_acc,
    "Test Accuracy": r.test_acc,
  }));

  const salaryData = (data.salary_comparison||[]).map(r=>({
    name: r.model,
    "CV Accuracy": r.cv_acc,
    "Test Accuracy": r.test_acc,
  }));

  const modelColors = ["#378ADD","#1D9E75","#D85A30","#7F77DD","#BA7517","#639922"];

  return (
    <div>
      <Card title="📊 Model Performance Comparison" subtitle="All algorithms trained and evaluated side by side">
        <div style={{ background:"#e6f1fb", borderRadius:10, padding:"10px 14px",
                      marginBottom:16, fontSize:13, color:"#0c447c" }}>
          <strong>Best placement model: {data.best_model}</strong> — auto-selected by highest 5-fold CV accuracy
        </div>

        <p style={{ fontSize:13, fontWeight:600, color:"#444", marginBottom:8 }}>
          Placement Predictor — 6 models compared
        </p>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={placementData} margin={{ top:4, right:20, bottom:4, left:0 }}>
            <XAxis dataKey="name" tick={{ fontSize:11 }}/>
            <YAxis domain={[50,100]} tick={{ fontSize:11 }} tickFormatter={v=>`${v}%`}/>
            <Tooltip formatter={v=>`${v}%`}/>
            <Legend wrapperStyle={{ fontSize:11 }}/>
            <Bar dataKey="CV Accuracy" radius={[4,4,0,0]}>
              {placementData.map((_,i)=><Cell key={i} fill={modelColors[i%modelColors.length]}/>)}
            </Bar>
            <Bar dataKey="Test Accuracy" fill="#ddd" radius={[4,4,0,0]}/>
          </BarChart>
        </ResponsiveContainer>

        <div style={{ marginTop:20 }}>
          <p style={{ fontSize:13, fontWeight:600, color:"#444", marginBottom:8 }}>
            Salary Tier Classifier — 4 models compared
          </p>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={salaryData} margin={{ top:4, right:20, bottom:4, left:0 }}>
              <XAxis dataKey="name" tick={{ fontSize:11 }}/>
              <YAxis domain={[40,80]} tick={{ fontSize:11 }} tickFormatter={v=>`${v}%`}/>
              <Tooltip formatter={v=>`${v}%`}/>
              <Legend wrapperStyle={{ fontSize:11 }}/>
              <Bar dataKey="CV Accuracy" radius={[4,4,0,0]}>
                {salaryData.map((_,i)=><Cell key={i} fill={modelColors[i%modelColors.length]}/>)}
              </Bar>
              <Bar dataKey="Test Accuracy" fill="#ddd" radius={[4,4,0,0]}/>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ marginTop:20 }}>
          <p style={{ fontSize:12, fontWeight:600, color:"#666", marginBottom:8 }}>
            Detailed results table:
          </p>
          <table style={{ width:"100%", borderCollapse:"collapse", fontSize:12 }}>
            <thead>
              <tr style={{ borderBottom:"1px solid #ddd" }}>
                <th style={{ textAlign:"left", padding:"6px 8px", color:"#888" }}>Model</th>
                <th style={{ textAlign:"right", padding:"6px 8px", color:"#888" }}>CV Accuracy</th>
                <th style={{ textAlign:"right", padding:"6px 8px", color:"#888" }}>Test Accuracy</th>
                <th style={{ textAlign:"right", padding:"6px 8px", color:"#888" }}>AUC</th>
                <th style={{ textAlign:"center", padding:"6px 8px", color:"#888" }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {(data.placement_comparison||[]).map((r,i)=>(
                <tr key={i} style={{ borderBottom:"0.5px solid #eee",
                                      background:r.model===data.best_model?"#f0f8ff":"" }}>
                  <td style={{ padding:"6px 8px", fontWeight:r.model===data.best_model?600:400 }}>
                    {r.model===data.best_model?"🏆 ":""}{r.model}
                  </td>
                  <td style={{ textAlign:"right", padding:"6px 8px" }}>{r.cv_acc}%</td>
                  <td style={{ textAlign:"right", padding:"6px 8px" }}>{r.test_acc}%</td>
                  <td style={{ textAlign:"right", padding:"6px 8px" }}>{r.auc||"—"}</td>
                  <td style={{ textAlign:"center", padding:"6px 8px" }}>
                    <span style={{ padding:"2px 8px", borderRadius:20, fontSize:10,
                                    background:r.model===data.best_model?"#eaf3de":"#f7f6f2",
                                    color:r.model===data.best_model?C.green:"#888" }}>
                      {r.model===data.best_model?"Best":"Tested"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

// ── Shared components ─────────────────────────────────────────────────────────
function Card({ title, subtitle, children }) {
  return (
    <div style={{ background:"#fff", borderRadius:16, padding:"1.5rem",
                  border:"1px solid #e8e6e0", marginBottom:20 }}>
      <h2 style={{ fontSize:17, fontWeight:700, margin:0 }}>{title}</h2>
      <p style={{ fontSize:12, color:"#888", marginTop:3, marginBottom:16 }}>{subtitle}</p>
      {children}
    </div>
  );
}
function Input({ label, type="text", value, onChange }) {
  return (
    <div style={{ marginBottom:12 }}>
      <label style={{ fontSize:12, color:"#666", display:"block", marginBottom:4 }}>{label}</label>
      <input type={type} value={value} onChange={e=>onChange(e.target.value)}
             style={{ width:"100%", padding:"9px 12px", borderRadius:8,
                      border:"1px solid #ddd", fontSize:13, boxSizing:"border-box" }}/>
    </div>
  );
}
function Select({ label, value, onChange, options }) {
  return (
    <div>
      <label style={{ fontSize:11, color:"#666", display:"block", marginBottom:3 }}>{label}</label>
      <select value={value} onChange={e=>onChange(e.target.value)}
              style={{ width:"100%", padding:"7px 10px", borderRadius:8,
                       border:"1px solid #ddd", fontSize:12 }}>
        {options.map(o=><option key={o.v} value={o.v}>{o.l}</option>)}
      </select>
    </div>
  );
}
function RangeInput({ label, value, onChange }) {
  return (
    <div>
      <label style={{ fontSize:11, color:"#666", display:"block", marginBottom:3 }}>
        {label}: <strong>{Math.round(value)}%</strong>
      </label>
      <input type="range" min={40} max={100} step={1} value={value}
             onChange={e=>onChange(e.target.value)} style={{ width:"100%" }}/>
    </div>
  );
}
function Btn({ children, onClick, color=C.blue, style={}, loading, type }) {
  return (
    <button onClick={onClick} type={type||"button"} disabled={loading}
            style={{ padding:"10px 20px", background:color, color:"#fff",
                     border:"none", borderRadius:8, fontSize:13, fontWeight:600,
                     cursor:loading?"not-allowed":"pointer",
                     opacity:loading?0.7:1, ...style }}>
      {loading?"Loading...":children}
    </button>
  );
}
function StatCard({ label, value, color }) {
  return (
    <div style={{ background:"#fafaf8", borderRadius:10, padding:"10px 14px",
                  borderTop:`3px solid ${color}` }}>
      <div style={{ fontSize:10, color:"#888", marginBottom:3 }}>{label}</div>
      <div style={{ fontSize:15, fontWeight:700, color }}>{value}</div>
    </div>
  );
}

export default App;
