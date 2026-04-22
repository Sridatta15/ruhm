import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time, os, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Smart Sericulture System",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (accuracy_score, mean_absolute_error,
                                  mean_squared_error, r2_score, confusion_matrix)
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

TEMP_MIN, TEMP_MAX = 24.0, 28.0
HUM_MIN,  HUM_MAX  = 70.0, 85.0
OPT_TMIN, OPT_TMAX = 25.0, 27.0
OPT_HMIN, OPT_HMAX = 80.0, 85.0

DARK_PLOT = {
    "figure.facecolor"  : "#1e2130",
    "axes.facecolor"    : "#252a3a",
    "axes.edgecolor"    : "#3a4060",
    "axes.labelcolor"   : "#c8cfe0",
    "xtick.color"       : "#8892b0",
    "ytick.color"       : "#8892b0",
    "text.color"        : "#c8cfe0",
    "axes.grid"         : True,
    "grid.color"        : "#2e3550",
    "grid.alpha"        : 0.8,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.spines.left"  : True,
    "axes.spines.bottom": True,
    "legend.facecolor"  : "#252a3a",
    "legend.edgecolor"  : "#3a4060",
    "legend.labelcolor" : "#c8cfe0",
}

st.markdown("""
<style>
/* ── Global background & font ───────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {
    background-color: #10131e !important;
    color: #c8cfe0 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141828 0%, #1a1f30 100%) !important;
    border-right: 1px solid #2a3050 !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Tabs ──────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #1a1f30 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid #2a3050 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8892b0 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: .84rem !important;
    padding: 8px 16px !important;
    transition: all .2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#2d6a4f,#1a472a) !important;
    color: #fff !important;
    box-shadow: 0 2px 12px rgba(45,106,79,.4) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 18px !important;
}

/* ── Streamlit widgets ──────────────────────── */
.stSlider [data-baseweb="slider"] { padding: 0 !important; }
.stSlider label, .stNumberInput label,
.stTextInput label, .stSelectbox label {
    color: #8892b0 !important; font-size:.82rem !important;
}
.stSlider [data-testid="stThumbValue"] { color:#64ffda !important; }
div[data-baseweb="input"] input,
div[data-baseweb="select"] { background:#1e2130 !important; color:#c8cfe0 !important; border-color:#2a3050 !important; }

/* st.metric */
[data-testid="metric-container"] {
    background: #1a1f30 !important;
    border: 1px solid #2a3050 !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}
[data-testid="metric-container"] label { color:#8892b0 !important; font-size:.8rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#e2e8f0 !important; font-weight:700 !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size:.8rem !important; }

/* st.dataframe */
[data-testid="stDataFrame"] { border-radius:10px !important; overflow:hidden !important; }
.stDataFrame thead tr th {
    background: #1e2130 !important; color:#64ffda !important;
    font-size:.8rem !important; font-weight:700 !important;
    border-bottom: 1px solid #2a3050 !important;
}
.stDataFrame tbody tr { background:#1a1f30 !important; }
.stDataFrame tbody tr:nth-child(even) { background:#1e2130 !important; }
.stDataFrame tbody td { color:#c8cfe0 !important; font-size:.8rem !important; border-color:#2a3050 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg,#2d6a4f,#1a472a) !important;
    color:#fff !important; border:none !important;
    border-radius:8px !important; font-weight:600 !important;
    transition: all .2s !important;
}
.stButton > button:hover { opacity:.88 !important; transform:translateY(-1px) !important; }

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg,#1a3a5c,#0d2137) !important;
    color:#64ffda !important; border:1px solid #1e4976 !important;
    border-radius:8px !important; font-weight:600 !important;
}

/* Progress bar */
.stProgress > div > div { background:linear-gradient(90deg,#2d6a4f,#64ffda) !important; border-radius:4px !important; }
.stProgress > div { background:#1e2130 !important; border-radius:4px !important; }

/* st.info / success / warning */
.stAlert { border-radius:10px !important; }
[data-testid="stInfo"]    { background:#0d2137 !important; border:1px solid #1e4976 !important; color:#64b5f6 !important; }
[data-testid="stSuccess"] { background:#0d2620 !important; border:1px solid #1a5c36 !important; color:#64ffda !important; }
[data-testid="stWarning"] { background:#2a1f00 !important; border:1px solid #6b4c00 !important; color:#ffd54f !important; }
[data-testid="stError"]   { background:#2a0a0a !important; border:1px solid #7f1d1d !important; color:#ff6b6b !important; }

/* Divider */
hr { border-color:#2a3050 !important; }

/* Sidebar divider */
[data-testid="stSidebar"] hr { border-color:#2a3050 !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown { color:#8892b0 !important; }

/* ── Custom components ───────────────────────── */

/* Header */
.dash-header {
    background: linear-gradient(135deg, #0d2a1a 0%, #1a472a 50%, #0d3348 100%);
    border: 1px solid #2d6a4f;
    border-radius: 16px;
    padding: 28px 36px;
    text-align: center;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.dash-header::before {
    content:'';
    position:absolute; top:0;left:0;right:0;bottom:0;
    background: radial-gradient(ellipse at 30% 50%, rgba(100,255,218,.06) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 50%, rgba(45,106,79,.08) 0%, transparent 60%);
}
.dash-header h1 {
    margin:0; font-size:2.1rem; font-weight:800;
    color:#ffffff; letter-spacing:-.5px;
    text-shadow: 0 0 30px rgba(100,255,218,.3);
}
.dash-header p {
    margin:6px 0 0; color:#8fb0a0; font-size:.9rem; letter-spacing:.3px;
}
.dash-header .badges { margin-top:12px; display:flex; justify-content:center; gap:8px; flex-wrap:wrap; }
.hbadge {
    display:inline-block; padding:3px 12px; border-radius:20px;
    font-size:.75rem; font-weight:700; letter-spacing:.5px;
}
.hb-green  { background:rgba(45,106,79,.3); color:#64ffda; border:1px solid rgba(100,255,218,.2); }
.hb-blue   { background:rgba(30,73,118,.3); color:#64b5f6; border:1px solid rgba(100,181,246,.2); }
.hb-purple { background:rgba(108,52,131,.3); color:#ce93d8; border:1px solid rgba(206,147,216,.2); }

/* Section header */
.sec-hdr {
    display:flex; align-items:center; gap:10px;
    font-size:1.15rem; font-weight:700; color:#e2e8f0;
    border-bottom:1px solid #2a3050;
    padding-bottom:10px; margin:12px 0 16px;
}
.sec-hdr .dot {
    width:8px; height:8px; border-radius:50%; flex-shrink:0;
}

/* Sensor cards */
.scard {
    background: linear-gradient(135deg, #1a1f30, #1e2436);
    border: 1px solid #2a3050;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 8px;
    position: relative;
    overflow: hidden;
    transition: border-color .2s;
}
.scard::before {
    content:''; position:absolute; top:0; left:0;
    width:4px; height:100%; border-radius:4px 0 0 4px;
}
.scard.green::before  { background:linear-gradient(180deg,#27ae60,#64ffda); }
.scard.red::before    { background:linear-gradient(180deg,#e74c3c,#ff6b6b); }
.scard.yellow::before { background:linear-gradient(180deg,#f39c12,#ffd54f); }
.scard.blue::before   { background:linear-gradient(180deg,#3498db,#64b5f6); }
.scard.purple::before { background:linear-gradient(180deg,#8e44ad,#ce93d8); }
.scard.teal::before   { background:linear-gradient(180deg,#16a085,#64ffda); }
.scard .slbl {
    font-size:.76rem; font-weight:600; color:#64b5f6;
    letter-spacing:.5px; text-transform:uppercase; margin-bottom:6px;
}
.scard .sval {
    font-size:1.85rem; font-weight:800; color:#e2e8f0;
    line-height:1.1; margin-bottom:4px;
}
.scard .ssub { font-size:.78rem; color:#5a6a8a; margin:1px 0; }
.scard .sdelta {
    font-size:.8rem; font-weight:700;
    display:inline-block; margin-top:4px;
    padding:2px 8px; border-radius:6px;
}
.sdelta.pos { background:rgba(39,174,96,.15); color:#64ffda; }
.sdelta.neg { background:rgba(231,76,60,.15);  color:#ff6b6b; }
.sdelta.neu { background:rgba(90,106,138,.15); color:#8892b0; }

/* ML card */
.mlcard {
    background: linear-gradient(135deg,#1a1530,#1e1a36);
    border:1px solid #3a2560;
    border-radius:12px; padding:16px 18px; margin-bottom:8px;
}
.mlcard::before { background:linear-gradient(180deg,#8e44ad,#ce93d8); }
.mlcard .slbl { color:#ce93d8 !important; }

/* Info box */
.ibox {
    background: linear-gradient(135deg,#0d2137,#0f2a42);
    border:1px solid #1e4976; border-left:3px solid #64b5f6;
    border-radius:10px; padding:12px 18px;
    font-size:.88rem; color:#90caf9; margin:8px 0;
}
.mlbox {
    background: linear-gradient(135deg,#1a1530,#1e1a36);
    border:1px solid #3a2560; border-left:3px solid #ce93d8;
    border-radius:10px; padding:12px 18px;
    font-size:.88rem; color:#e0b4ff; margin:8px 0;
}
.warnbox {
    background: linear-gradient(135deg,#2a1a00,#2e1f00);
    border:1px solid #6b4c00; border-left:3px solid #ffd54f;
    border-radius:10px; padding:12px 18px;
    font-size:.88rem; color:#ffd54f; margin:8px 0;
}
.critbox {
    background: linear-gradient(135deg,#2a0a0a,#2e0e0e);
    border:2px solid #ff6b6b; border-radius:10px;
    padding:12px 18px; color:#ff6b6b;
    font-weight:700; font-size:.95rem; text-align:center; margin:8px 0;
    animation: pulse-red 1.5s ease-in-out infinite alternate;
}
@keyframes pulse-red {
    from { box-shadow:0 0 8px rgba(231,76,60,.3); }
    to   { box-shadow:0 0 20px rgba(231,76,60,.6); }
}

/* Intelligence panel */
.ipanel {
    border-radius:12px; padding:16px 18px; margin-top:8px;
    font-size:.88rem; border:1px solid #2a3050;
}
.ipanel.opt { background:linear-gradient(135deg,#0a2a1a,#0d3020); border-color:#1a5c36; }
.ipanel.wrn { background:linear-gradient(135deg,#2a1f00,#2e2400); border-color:#6b4c00; }
.ipanel.crt { background:linear-gradient(135deg,#2a0a0a,#2e0e0e); border-color:#7f1d1d; }

/* Badges */
.sbadge { display:inline-block; padding:4px 14px; border-radius:20px; font-weight:700; font-size:.8rem; }
.sopt   { background:rgba(39,174,96,.2); color:#64ffda; border:1px solid rgba(100,255,218,.3); }
.swrn   { background:rgba(243,156,18,.2); color:#ffd54f; border:1px solid rgba(255,213,79,.3); }
.scrt   { background:rgba(231,76,60,.2);  color:#ff6b6b; border:1px solid rgba(255,107,107,.3); }

.dtag { display:inline-block; padding:3px 10px; border-radius:8px; font-size:.78rem; font-weight:600; margin:2px; }
.dsafe { background:rgba(39,174,96,.15); color:#64ffda; border:1px solid rgba(100,255,218,.2); }
.drisk { background:rgba(231,76,60,.15); color:#ff6b6b; border:1px solid rgba(255,107,107,.2); }

.chip   { display:inline-block; padding:3px 10px; border-radius:8px;
          background:rgba(30,73,118,.3); color:#64b5f6;
          font-size:.76rem; font-weight:600; margin:2px; border:1px solid rgba(100,181,246,.2); }
.mlchip { display:inline-block; padding:3px 10px; border-radius:8px;
          background:rgba(108,52,131,.25); color:#ce93d8;
          font-size:.76rem; font-weight:600; margin:2px; border:1px solid rgba(206,147,216,.2); }

/* Action badge */
.abadge { display:inline-block; padding:8px 20px; border-radius:20px; font-weight:700; font-size:.95rem; }
.afan   { background:rgba(30,73,118,.3); color:#64b5f6; border:1px solid rgba(100,181,246,.3); }
.aheat  { background:rgba(120,66,18,.3); color:#ffb74d; border:1px solid rgba(255,183,77,.3); }
.ahum   { background:rgba(39,174,96,.2); color:#64ffda; border:1px solid rgba(100,255,218,.2); }
.asta   { background:rgba(39,174,96,.15); color:#a8e6cf; border:1px solid rgba(168,230,207,.2); }
.amul   { background:rgba(231,76,60,.2); color:#ff6b6b; border:1px solid rgba(255,107,107,.3); }

/* Health bar */
.hbar-wrap { background:#1e2130; border-radius:8px; height:10px; overflow:hidden; margin:6px 0; }
.hbar-fill { height:10px; border-radius:8px; transition:width .4s ease; }

/* KPI strip */
.kpi-strip {
    display:grid; gap:12px;
    grid-template-columns: repeat(auto-fit,minmax(140px,1fr));
    margin:12px 0;
}
.kpi {
    background:linear-gradient(135deg,#1a1f30,#1e2436);
    border:1px solid #2a3050; border-radius:12px;
    padding:14px 16px; text-align:center;
}
.kpi .kv { font-size:1.6rem; font-weight:800; color:#e2e8f0; line-height:1; }
.kpi .kl { font-size:.74rem; color:#5a6a8a; margin-top:4px; font-weight:600; text-transform:uppercase; letter-spacing:.5px; }
.kpi .ks { font-size:.78rem; font-weight:700; margin-top:2px; }

/* Step ticker */
.step-ticker {
    background:linear-gradient(135deg,#1a1f30,#1e2436);
    border:1px solid #2a3050; border-radius:10px;
    padding:10px 18px; font-size:.88rem; color:#8892b0;
    display:flex; align-items:center; gap:8px; margin:8px 0;
}
.step-ticker .pulse {
    width:8px;height:8px;border-radius:50%;background:#64ffda;
    animation:blink 1s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:.2;} }

/* Table section */
.tsec {
    background:linear-gradient(135deg,#1a1f30,#1e2436);
    border:1px solid #2a3050; border-radius:12px; padding:18px; margin:12px 0;
}
.tsec-title { font-size:.9rem; font-weight:700; color:#64b5f6; margin-bottom:12px; text-transform:uppercase; letter-spacing:.5px; }

/* Sidebar stat */
.sstat {
    background:#1a1f30; border:1px solid #2a3050; border-radius:8px;
    padding:8px 12px; margin:4px 0; display:flex; justify-content:space-between;
}
.sstat .sk { font-size:.78rem; color:#5a6a8a; }
.sstat .sv { font-size:.82rem; font-weight:700; color:#64ffda; }

/* color utilities */
.cg { color:#64ffda; } .cy { color:#ffd54f; } .cr { color:#ff6b6b; }
.cb { color:#64b5f6; } .cp { color:#ce93d8; }
</style>
""", unsafe_allow_html=True)


def sec(icon, title, color="#64ffda"):
    st.markdown(
        f"<div class='sec-hdr'>"
        f"<div class='dot' style='background:{color};box-shadow:0 0 8px {color}88;'></div>"
        f"{icon} {title}</div>",
        unsafe_allow_html=True)

def scard(label, value, sub1="", sub2="", color="blue", delta=None):
    dc = ""
    if delta is not None:
        sign = "pos" if delta > 0 else ("neg" if delta < 0 else "neu")
        dc = f"<div class='sdelta {sign}'>{delta:+.2f}</div>"
    return (
        f"<div class='scard {color}'>"
        f"<div class='slbl'>{label}</div>"
        f"<div class='sval'>{value}</div>"
        f"{'<div class=ssub>'+sub1+'</div>' if sub1 else ''}"
        f"{'<div class=ssub>'+sub2+'</div>' if sub2 else ''}"
        f"{dc}</div>"
    )

def mlcard(label, value, sub=""):
    return (
        f"<div class='scard purple mlcard'>"
        f"<div class='slbl' style='color:#ce93d8;'>{label}</div>"
        f"<div class='sval' style='color:#e2e8f0;'>{value}</div>"
        f"{'<div class=ssub style=color:#8060a0;>'+sub+'</div>' if sub else ''}"
        f"</div>"
    )

def action_badge(a):
    a = str(a)
    cls = ("afan" if "Fan" in a else "aheat" if "Heater" in a
           else "ahum" if "Humidif" in a else "amul" if "Multi" in a else "asta")
    icons = {"Fan":"🌬","Heater":"🔥","Humidif":"💧","Dehumid":"💨","Multi":"⚡","Stable":"✅"}
    ic = next((v for k,v in icons.items() if k in a), "⚙️")
    return f'<span class="abadge {cls}">{ic} {a}</span>'

def status_info(temp, hum):
    td = max(0, OPT_TMIN-temp if temp<OPT_TMIN else temp-OPT_TMAX)
    hd = max(0, OPT_HMIN-hum  if hum <OPT_HMIN else hum -OPT_HMAX)
    s  = "Optimal" if td==0 and hd==0 else ("Warning" if td<=2 and hd<=5 else "Critical")
    sc = round(max(0, 100-min(100, td*8+hd*2)), 1)
    return s, sc, round(td,3), round(hd,3)

def disease_risk_str(t,h):
    r = []
    if t > OPT_TMAX: r.append("Flacherie")
    if t < OPT_TMIN: r.append("Grasserie")
    if h > OPT_HMAX: r.append("Muscardine")
    return " | ".join(r) if r else "Safe"

def suggestions_str(t,h):
    s = []
    if t > OPT_TMAX: s.append("Turn ON AC")
    if t < OPT_TMIN: s.append("Turn ON Heater")
    if h < OPT_HMIN: s.append("Turn ON Humidifier")
    if h > OPT_HMAX: s.append("Turn ON Dehumidifier")
    return " | ".join(s) if s else "No action needed"

def intel_panel_html(temp, hum, hist_t=None):
    s, sc, td, hd = status_info(temp, hum)
    risks  = disease_risk_str(temp, hum)
    suggs  = suggestions_str(temp, hum)
    pcls   = {"Optimal":"opt","Warning":"wrn","Critical":"crt"}[s]
    bcls   = {"Optimal":"sopt","Warning":"swrn","Critical":"scrt"}[s]
    icon   = {"Optimal":"🟢","Warning":"🟡","Critical":"🔴"}[s]
    bc     = "#64ffda" if sc>=70 else ("#ffd54f" if sc>=40 else "#ff6b6b")
    d_html = "".join(
        f'<span class="dtag {"dsafe" if r.strip()=="Safe" else "drisk"}">'
        f'{"✅" if r.strip()=="Safe" else "⚠"} {r.strip()}</span>'
        for r in risks.split("|")
    )
    s_html = "".join(f'<span class="chip">💡 {s.strip()}</span>' for s in suggs.split("|"))
    pred   = ""
    if hist_t and len(hist_t)>=5:
        p = round(sum(hist_t[-5:])/5,2)
        pred = f'<div style="font-size:.76rem;color:#5a6a8a;margin-top:6px;">📈 MA5 Forecast: <b style="color:{bc};">{p}°C</b></div>'
    return (
        f"<div class='ipanel {pcls}'>"
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;'>"
        f"<span class='sbadge {bcls}'>{icon} {s}</span>"
        f"<span style='font-size:.76rem;color:#5a6a8a;'>T-dev {td:.1f}°C · H-dev {hd:.1f}%</span></div>"
        f"<div style='margin-bottom:6px;font-size:.82rem;'><b style='color:#8892b0;'>Disease Risk:</b> {d_html}</div>"
        f"<div style='margin-bottom:8px;font-size:.82rem;'><b style='color:#8892b0;'>Action:</b> {s_html}</div>"
        f"<div style='font-size:.74rem;color:#5a6a8a;margin-bottom:3px;'>System Health Score</div>"
        f"<div class='hbar-wrap'><div class='hbar-fill' style='width:{sc}%;background:linear-gradient(90deg,{bc}88,{bc});'></div></div>"
        f"<div style='font-size:.85rem;font-weight:700;color:{bc};'>{sc} / 100</div>"
        f"{pred}</div>"
    )

def determine_action(temp, hum):
    a = []
    if temp>TEMP_MAX: a.append("Fan ON")
    if temp<TEMP_MIN: a.append("Heater ON")
    if hum <HUM_MIN:  a.append("Humidifier ON")
    if hum >HUM_MAX:  a.append("Dehumidifier ON")
    if len(a)>=2: return "Multi: "+" + ".join(a)
    return a[0] if a else "Stable"


ENHANCED_PATH = "weatherHistory_enhanced.csv"
FALLBACK_PATH = "weatherHistory.csv"

@st.cache_data(show_spinner="📂 Loading enhanced dataset…")
def load_enhanced(path):
    df = pd.read_csv(path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    num_cols = ["temperature","apparent_temp","humidity","wind_speed","wind_bearing",
                "visibility","pressure","controlled_temp","controlled_humidity",
                "health_score","dew_point","heat_index","wind_chill",
                "temp_lag1","temp_lag2","temp_lag3",
                "humidity_lag1","humidity_lag2","humidity_lag3",
                "wind_lag1","pres_lag1",
                "temp_roll3_mean","temp_roll3_std","temp_roll6_mean","temp_roll6_std",
                "hum_roll3_mean","hum_roll3_std","hum_roll6_mean","hum_roll6_std",
                "next_temp","next_humidity","hour_sin","hour_cos","month_sin","month_cos",
                "temp_diff_apparent","temp_humidity_index","temp_deviation","humidity_deviation",
                "wind_roll6_mean","pres_roll6_mean","is_daytime","silkworm_suitable",
                "temp_in_range","humidity_in_range","ctrl_temp_in_range","ctrl_hum_in_range",
                "temp_correction_delta","hum_correction_delta"]
    for c in num_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["precip_type"]  = df["precip_type"].fillna("None")
    df["action_label"] = df["action_label"].fillna("Stable")
    return df

FEAT_COLS = [
    "temperature","humidity","apparent_temp","wind_speed","wind_bearing",
    "visibility","pressure","dew_point","heat_index","wind_chill",
    "temp_diff_apparent","temp_humidity_index",
    "temp_lag1","temp_lag2","temp_lag3",
    "humidity_lag1","humidity_lag2","humidity_lag3",
    "wind_lag1","pres_lag1",
    "temp_roll3_mean","temp_roll3_std","temp_roll6_mean","temp_roll6_std",
    "hum_roll3_mean","hum_roll3_std","hum_roll6_mean","hum_roll6_std",
    "wind_roll6_mean","pres_roll6_mean",
    "hour_sin","hour_cos","month_sin","month_cos",
    "is_daytime","health_score","temp_deviation","humidity_deviation",
]

@st.cache_data(show_spinner="🤖 Training ML models…")
def train_models(df):
    if not SKLEARN_OK: return None
    fa = [c for c in FEAT_COLS if c in df.columns]
    d  = df.dropna(subset=fa+["next_temp","next_humidity","action_label"]).copy()
    if len(d)<100: return None
    for c in fa+["next_temp","next_humidity"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=fa+["next_temp","next_humidity"])
    X  = d[fa].values.astype(float)
    ya = d["action_label"].values
    yt = d["next_temp"].values.astype(float)
    yh = d["next_humidity"].values.astype(float)
    le = LabelEncoder(); ya_enc = le.fit_transform(ya)
    sp = int(len(d)*0.8)
    Xtr,Xte       = X[:sp],      X[sp:]
    yatr,yate     = ya_enc[:sp], ya_enc[sp:]
    yttr,ytte     = yt[:sp],     yt[sp:]
    yhtr,yhte     = yh[:sp],     yh[sp:]
    clf = RandomForestClassifier(n_estimators=100,max_depth=12,n_jobs=-1,random_state=42)
    clf.fit(Xtr,yatr)
    ya_pred = clf.predict(Xte)
    acc = accuracy_score(yate,ya_pred)
    cm  = confusion_matrix(yate,ya_pred)
    rfr = RandomForestRegressor(n_estimators=100,max_depth=12,n_jobs=-1,random_state=42)
    rfr.fit(Xtr,yttr); yt_pred=rfr.predict(Xte)
    t_mae=mean_absolute_error(ytte,yt_pred)
    t_rmse=np.sqrt(mean_squared_error(ytte,yt_pred))
    t_r2=r2_score(ytte,yt_pred)
    rid = Ridge(alpha=1.0); rid.fit(Xtr,yhtr); yh_pred=rid.predict(Xte)
    h_mae=mean_absolute_error(yhte,yh_pred)
    h_rmse=np.sqrt(mean_squared_error(yhte,yh_pred))
    h_r2=r2_score(yhte,yh_pred)
    return dict(clf=clf,rfr=rfr,rid=rid,le=le,feat_cols=fa,classes=le.classes_,
                acc=acc,cm=cm,t_mae=t_mae,t_rmse=t_rmse,t_r2=t_r2,
                h_mae=h_mae,h_rmse=h_rmse,h_r2=h_r2,
                ytte=ytte,yt_pred=yt_pred,yhte=yhte,yh_pred=yh_pred,
                yate=yate,ya_pred=ya_pred,
                imp_clf=clf.feature_importances_,imp_rfr=rfr.feature_importances_,
                n_train=len(Xtr),n_test=len(Xte))

def live_infer(models, rec):
    if models is None: return None,None,None
    x = np.array([float(rec.get(f,0) or 0) for f in models["feat_cols"]], dtype=float)
    if np.any(np.isnan(x)): return None,None,None
    X = x.reshape(1,-1)
    pa = models["le"].inverse_transform(models["clf"].predict(X))[0]
    pt = round(float(models["rfr"].predict(X)[0]),2)
    ph = round(float(models["rid"].predict(X)[0]),2)
    return pa, pt, ph

def rolling_forecast(models, df_live, horizon=15):
    if models is None or len(df_live)<4: return [],[]
    row = df_live.iloc[-1].copy()
    tf,hf = [],[]
    for _ in range(horizon):
        x = np.array([row[f] if f in row.index and pd.notna(row[f]) else 0.0
                      for f in models["feat_cols"]], dtype=float)
        if np.any(np.isnan(x)): break
        pt = float(models["rfr"].predict(x.reshape(1,-1))[0])
        ph = float(models["rid"].predict(x.reshape(1,-1))[0])
        tf.append(round(pt,2)); hf.append(round(ph,2))
        for i in range(3,1,-1):
            if f"temp_lag{i}"     in row.index: row[f"temp_lag{i}"]     = row.get(f"temp_lag{i-1}",pt)
            if f"humidity_lag{i}" in row.index: row[f"humidity_lag{i}"] = row.get(f"humidity_lag{i-1}",ph)
        if "temp_lag1"      in row.index: row["temp_lag1"]      = row["temperature"]
        if "humidity_lag1"  in row.index: row["humidity_lag1"]  = row["humidity"]
        if "temperature"    in row.index: row["temperature"]    = pt
        if "humidity"       in row.index: row["humidity"]       = ph
        if "temp_roll3_mean" in row.index:
            row["temp_roll3_mean"] = np.mean([row.get("temp_lag1",pt),row.get("temp_lag2",pt),pt])
        if "hum_roll3_mean" in row.index:
            row["hum_roll3_mean"]  = np.mean([row.get("humidity_lag1",ph),row.get("humidity_lag2",ph),ph])
        _,sc,td,hd = status_info(pt,ph)
        for k,v in [("health_score",sc),("temp_deviation",td),("humidity_deviation",hd)]:
            if k in row.index: row[k]=v
    return tf, hf


for k,v in dict(rows=[],ml_pa=[],ml_pt=[],ml_ph=[]).items():
    if k not in st.session_state: st.session_state[k]=[]

def reset_state():
    for k in ["rows","ml_pa","ml_pt","ml_ph"]: st.session_state[k]=[]

def get_ldf():
    if not st.session_state["rows"]: return None
    return pd.DataFrame(st.session_state["rows"])


data_path = ENHANCED_PATH if os.path.exists(ENHANCED_PATH) else FALLBACK_PATH
if not os.path.exists(data_path):
    st.error(f"❌ Dataset not found: `{data_path}`"); st.stop()

df_full = load_enhanced(data_path)
ml      = train_models(df_full) if SKLEARN_OK else None


with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px;'>
      <div style='font-size:2.5rem;'>🐛</div>
      <div style='font-size:1rem;font-weight:800;color:#e2e8f0;letter-spacing:-.3px;'>Smart Sericulture</div>
      <div style='font-size:.75rem;color:#5a6a8a;margin-top:2px;'>IoT + AI Environment System</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("<div style='font-size:.78rem;font-weight:700;color:#64b5f6;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;'>🌡 Optimal Ranges</div>", unsafe_allow_html=True)
    TEMP_MIN = st.slider("Min Temp °C",  18.0, 28.0, 24.0, 0.5)
    TEMP_MAX = st.slider("Max Temp °C",  24.0, 35.0, 28.0, 0.5)
    HUM_MIN  = st.slider("Min Humidity%",40.0, 75.0, 70.0, 1.0)
    HUM_MAX  = st.slider("Max Humidity%",75.0, 95.0, 85.0, 1.0)
    st.divider()

    st.markdown("<div style='font-size:.78rem;font-weight:700;color:#64b5f6;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;'>⚙️ Simulation</div>", unsafe_allow_html=True)
    sim_speed  = st.slider("Speed s/step", 0.05, 2.0, 0.15, 0.05)
    sim_steps  = st.number_input("Steps", 20, 5000, 200, 50)
    ml_horizon = st.slider("🔮 Forecast Steps", 5, 60, 20, 5)
    st.divider()

    if st.button("🔄 Reset Simulation", use_container_width=True):
        reset_state(); st.rerun()

    n_done = len(st.session_state["rows"])
    st.divider()

    st.markdown("<div style='font-size:.78rem;font-weight:700;color:#64b5f6;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;'>📊 Status</div>", unsafe_allow_html=True)
    if n_done:
        st.markdown(f"<div class='sstat'><span class='sk'>Steps done</span><span class='sv'>{n_done:,}</span></div>", unsafe_allow_html=True)
    if SKLEARN_OK and ml:
        st.markdown(f"<div class='sstat'><span class='sk'>Classifier Acc</span><span class='sv'>{ml['acc']*100:.2f}%</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sstat'><span class='sk'>Temp R²</span><span class='sv'>{ml['t_r2']:.4f}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sstat'><span class='sk'>Humidity R²</span><span class='sv'>{ml['h_r2']:.4f}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sstat'><span class='sk'>Train rows</span><span class='sv'>{ml['n_train']:,}</span></div>", unsafe_allow_html=True)
    elif not SKLEARN_OK:
        st.warning("Install scikit-learn")

    st.divider()
    st.markdown(f"""<div style='font-size:.74rem;color:#3a4a6a;text-align:center;line-height:1.6;'>
    weatherHistory_enhanced.csv<br>
    {len(df_full):,} rows · 76 columns<br>
    VIT Vellore · 2025–26
    </div>""", unsafe_allow_html=True)


st.markdown("""
<div class='dash-header'>
  <h1>🐛 Smart Sericulture Environment System</h1>
  <p>Simulation-Based IoT · AI Intelligence · Machine Learning Predictions · Automated Control</p>
  <div class='badges'>
    <span class='hbadge hb-green'>96,429 Rows</span>
    <span class='hbadge hb-green'>76 Features</span>
    <span class='hbadge hb-blue'>3 ML Models</span>
    <span class='hbadge hb-blue'>Live Simulation</span>
    <span class='hbadge hb-purple'>RF Classifier 99.99%</span>
    <span class='hbadge hb-purple'>Temp R² 0.9881</span>
    <span class='hbadge hb-green'>VIT Vellore 2025-26</span>
  </div>
</div>
""", unsafe_allow_html=True)

t1,t2,t3,t4,t5,t6,t7 = st.tabs([
    "🔴 Live Simulation","📊 Data Analysis","📈 Visualizations",
    "⚙️ Control Results","🧠 AI Intelligence","🤖 ML Predictions","📖 About",
])


with t1:
    sec("🔴","Real-Time Sensor Simulation","#ff6b6b")
    st.markdown("""<div class='ibox'>
    Streams <b>weatherHistory_enhanced.csv</b> row-by-row — all 76 pre-engineered columns captured per step.
    <b>ML models</b> (RF Classifier + RF Regressor + Ridge) run live inference every step.
    Purple dashed lines on charts = ML predictions overlaid on actual values.
    </div>""", unsafe_allow_html=True)

    cb1,cb2,cb3 = st.columns([1,1,5])
    start = cb1.button("▶ Start Simulation", type="primary")
    stop  = cb2.button("⏹ Stop")
    cb3.markdown(
        f"<div style='padding-top:8px;font-size:.82rem;color:#5a6a8a;'>"
        f"Dataset: <b style='color:#64b5f6;'>{len(df_full):,}</b> rows · "
        f"<b style='color:#64b5f6;'>{int(sim_steps)}</b> steps · "
        f"<b style='color:#64b5f6;'>{sim_speed}s</b>/step</div>",
        unsafe_allow_html=True)

    ph_step = st.empty()
    ph_prog = st.empty()

    st.markdown("<div class='sec-hdr' style='margin-top:14px;'><div class='dot' style='background:#64ffda;'></div>🌡 Core Sensors</div>", unsafe_allow_html=True)
    r1 = st.columns(4)
    ph_t,ph_h,ph_ct,ph_ch = r1[0].empty(),r1[1].empty(),r1[2].empty(),r1[3].empty()

    st.markdown("<div class='sec-hdr'><div class='dot' style='background:#64b5f6;'></div>🌍 Environmental Context</div>", unsafe_allow_html=True)
    r2 = st.columns(5)
    ph_ws,ph_pr,ph_vi,ph_dp,ph_hi = (r2[i].empty() for i in range(5))
    r3 = st.columns(3)
    ph_wc,ph_pt2,ph_ss = r3[0].empty(),r3[1].empty(),r3[2].empty()

    st.markdown("<div class='sec-hdr'><div class='dot' style='background:#ffb74d;'></div>⚙️ Control & Intelligence</div>", unsafe_allow_html=True)
    ph_act   = st.empty()
    ph_intel = st.empty()

    st.markdown("<div class='sec-hdr'><div class='dot' style='background:#ce93d8;'></div>🤖 ML Live Inference</div>", unsafe_allow_html=True)
    r4 = st.columns(3)
    ph_mla,ph_mlt,ph_mlh = r4[0].empty(),r4[1].empty(),r4[2].empty()

    ph_chart = st.empty()

    if start:
        reset_state()
        step_df = df_full.head(int(sim_steps)).reset_index(drop=True)

        for _, row in step_df.iterrows():
            if stop: break
            rec = row.to_dict()
            t   = float(rec.get("temperature",20) or 20)
            h   = float(rec.get("humidity",75) or 75)
            ct  = float(rec.get("controlled_temp",t) or t)
            ch  = float(rec.get("controlled_humidity",h) or h)
            act = str(rec.get("action","Stable"))
            at  = float(rec.get("apparent_temp",t) or t)
            ws  = float(rec.get("wind_speed",0) or 0)
            pr  = float(rec.get("pressure",1013) or 1013)
            vi  = float(rec.get("visibility",10) or 10)
            dp  = float(rec.get("dew_point",t) or t)
            hi_ = float(rec.get("heat_index",t) or t)
            wc  = float(rec.get("wind_chill",t) or t)
            pt2 = str(rec.get("precip_type","None"))
            ts  = rec.get("timestamp","")
            hs  = float(rec.get("health_score",0) or 0)
            ist = str(rec.get("intel_status","Critical"))
            dr  = str(rec.get("disease_risk","Unknown"))
            ss  = str(rec.get("smart_suggestion",""))
            sc_ = str(rec.get("season",""))
            wcat= str(rec.get("wind_category",""))
            vcat= str(rec.get("visibility_category",""))

            st.session_state["rows"].append(rec)
            pa, pt_, ph_ = live_infer(ml, rec)
            st.session_state["ml_pa"].append(pa)
            st.session_state["ml_pt"].append(pt_)
            st.session_state["ml_ph"].append(ph_)

            n = len(st.session_state["rows"])
            ts_str = str(ts)[:19]
            s_,sc_val,_,_ = status_info(t,h)
            s_color = {"Optimal":"#64ffda","Warning":"#ffd54f","Critical":"#ff6b6b"}[s_]

            ph_step.markdown(
                f"<div class='step-ticker'><div class='pulse' style='background:{s_color};box-shadow:0 0 8px {s_color};'></div>"
                f"<span style='color:#8892b0;'>Step</span> "
                f"<b style='color:#e2e8f0;'>{n}</b>"
                f"<span style='color:#3a4a6a;'>/</span>"
                f"<span style='color:#5a6a8a;'>{int(sim_steps)}</span>"
                f"&nbsp;&nbsp;<span style='color:#3a4a6a;'>|</span>&nbsp;&nbsp;"
                f"<span style='color:#5a6a8a;'>{ts_str}</span>"
                f"&nbsp;&nbsp;<span style='color:#3a4a6a;'>|</span>&nbsp;&nbsp;"
                f"<span style='color:{s_color};font-weight:700;'>{s_}</span>"
                f"&nbsp;&nbsp;<span style='color:#3a4a6a;'>|</span>&nbsp;&nbsp;"
                f"<span style='color:#5a6a8a;'>Season: </span><b style='color:#64b5f6;'>{sc_}</b>"
                f"</div>",
                unsafe_allow_html=True)
            ph_prog.progress(n/int(sim_steps))

            t_col = "green" if TEMP_MIN<=t<=TEMP_MAX else ("yellow" if abs(t-26)<4 else "red")
            h_col = "green" if HUM_MIN<=h<=HUM_MAX else "yellow"

            ph_t.markdown(scard("🌡 RAW TEMPERATURE",f"{t:.2f}°C",
                f"Feels like {at:.1f}°C",f"Range {TEMP_MIN}–{TEMP_MAX}°C",t_col,ct-t), unsafe_allow_html=True)
            ph_h.markdown(scard("💧 RAW HUMIDITY",f"{h:.1f}%",
                f"Range {HUM_MIN}–{HUM_MAX}%","",h_col,ch-h), unsafe_allow_html=True)
            ph_ct.markdown(scard("✅ CONTROLLED TEMP",f"{ct:.2f}°C",
                f"Δ {ct-t:+.2f}°C","","teal"), unsafe_allow_html=True)
            ph_ch.markdown(scard("✅ CONTROLLED HUM",f"{ch:.1f}%",
                f"Δ {ch-h:+.2f}%","","teal"), unsafe_allow_html=True)

            ph_ws.markdown(scard("🌬 WIND SPEED",f"{ws:.1f}",f"km/h · {wcat}","","blue"), unsafe_allow_html=True)
            ph_pr.markdown(scard("🔵 PRESSURE",f"{pr:.0f}","mb","","blue"), unsafe_allow_html=True)
            ph_vi.markdown(scard("👁 VISIBILITY",f"{vi:.1f}",f"km · {vcat}","","blue"), unsafe_allow_html=True)
            ph_dp.markdown(scard("💦 DEW POINT",f"{dp:.1f}°C","","","teal"), unsafe_allow_html=True)
            ph_hi.markdown(scard("🔥 HEAT INDEX",f"{hi_:.1f}°C","","","yellow"), unsafe_allow_html=True)
            ph_wc.markdown(scard("❄ WIND CHILL",f"{wc:.1f}°C","","","blue"), unsafe_allow_html=True)
            ph_pt2.markdown(scard("🌧 PRECIP TYPE",pt2,"","","blue"), unsafe_allow_html=True)
            ph_ss.markdown(
                f"<div class='scard blue'><div class='slbl'>🧠 SMART SUGGESTION</div>"
                f"<div style='font-size:.85rem;color:#90caf9;margin-top:4px;font-weight:600;'>{ss}</div>"
                f"<div class='ssub' style='color:#3a5070;margin-top:4px;'>Disease: {dr}</div></div>",
                unsafe_allow_html=True)

            alert = (f"<div class='critbox'>🚨 CRITICAL — UNSAFE ENVIRONMENT FOR SILKWORMS! Immediate action required!</div>"
                     if ist=="Critical" else "")
            ph_act.markdown(
                f"<div style='text-align:center;margin:10px 0;'>{action_badge(act)}"
                f"&nbsp;&nbsp;<span style='font-size:.82rem;color:#5a6a8a;'>"
                f"Health: <b style='color:{s_color};'>{hs:.0f}/100</b> · "
                f"<b style='color:{s_color};'>{ist}</b></span></div>",
                unsafe_allow_html=True)
            ph_intel.markdown(
                alert + intel_panel_html(t,h,[r["temperature"] for r in st.session_state["rows"]]),
                unsafe_allow_html=True)

            if pt_ is not None:
                ph_mla.markdown(mlcard("🤖 ML PREDICTED ACTION",pa,"RF Classifier · 100 trees"), unsafe_allow_html=True)
                ph_mlt.markdown(mlcard("🌡 ML NEXT TEMPERATURE",f"{pt_:.2f}°C",f"Actual: {t:.2f}°C · Δ {pt_-t:+.2f}°C"), unsafe_allow_html=True)
                ph_mlh.markdown(mlcard("💧 ML NEXT HUMIDITY",f"{ph_:.1f}%",f"Actual: {h:.1f}% · Δ {ph_-h:+.2f}%"), unsafe_allow_html=True)
            else:
                ph_mla.markdown("<div class='scard purple' style='color:#8060a0;font-size:.82rem;padding:12px;'>🤖 Warming up — need ≥4 steps…</div>", unsafe_allow_html=True)

            if n >= 3:
                ldf_ = get_ldf()
                xs = list(range(n))
                with plt.rc_context(DARK_PLOT):
                    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(13,3.2))
                    ax1.plot(xs, ldf_["temperature"],     color="#ff6b6b",lw=1.0,label="Raw Temp",alpha=0.9)
                    ax1.plot(xs, ldf_["controlled_temp"], color="#64ffda",lw=1.3,label="Controlled")
                    ml_tp=[v for v in st.session_state["ml_pt"] if v is not None]
                    if ml_tp:
                        xs_ml=[i for i,v in enumerate(st.session_state["ml_pt"]) if v is not None]
                        ax1.plot(xs_ml,ml_tp,color="#ce93d8",lw=0.9,ls="--",alpha=0.9,label="ML Pred")
                    ax1.axhspan(TEMP_MIN,TEMP_MAX,color="#64ffda",alpha=0.06)
                    ax1.set_title("Temperature (°C)",color="#c8cfe0",fontweight="bold")
                    ax1.legend(fontsize=8); ax1.set_xlabel("Step")

                    ax2.plot(xs, ldf_["humidity"],            color="#64b5f6",lw=1.0,label="Raw Hum",alpha=0.9)
                    ax2.plot(xs, ldf_["controlled_humidity"], color="#64ffda",lw=1.3,label="Controlled")
                    ml_hp=[v for v in st.session_state["ml_ph"] if v is not None]
                    if ml_hp:
                        xs_mh=[i for i,v in enumerate(st.session_state["ml_ph"]) if v is not None]
                        ax2.plot(xs_mh,ml_hp,color="#ce93d8",lw=0.9,ls="--",alpha=0.9,label="ML Pred")
                    ax2.axhspan(HUM_MIN,HUM_MAX,color="#64ffda",alpha=0.06)
                    ax2.set_title("Humidity (%)",color="#c8cfe0",fontweight="bold")
                    ax2.legend(fontsize=8); ax2.set_xlabel("Step")
                    plt.tight_layout(pad=1.5)
                ph_chart.pyplot(fig); plt.close(fig)

            time.sleep(sim_speed)

        ph_step.success("✅ Simulation complete! Explore all 7 tabs for full analysis.")


with t2:
    sec("📊","Full Dataset Analysis","#64b5f6")

    tot = len(df_full)
    st.markdown("<div class='kpi-strip'>"+
        "".join(f"<div class='kpi'><div class='kv'>{v}</div><div class='kl'>{l}</div></div>" for v,l in [
            (f"{tot:,}","Total Rows"),
            ("76","Features"),
            (f"{df_full['year'].nunique()}","Years"),
            (f"{df_full['silkworm_suitable'].sum():,}","Silkworm Suitable"),
            (f"{(df_full['intel_status']=='Optimal').sum():,}","Optimal Steps"),
            (f"{(df_full['intel_status']=='Critical').sum():,}","Critical Steps"),
        ])+"</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📋 Pre-built Feature Groups (76 columns)")
    col_groups = {
        "🕐 Time (12)"        :["hour","day_of_week","month","year","season","is_daytime","hour_sin","hour_cos","month_sin","month_cos","day_of_week_name","month_name"],
        "⚙️ Control (5)"      :["action","action_label","controlled_temp","controlled_humidity","temp_correction_delta"],
        "🧠 Intelligence (6)" :["intel_status","health_score","temp_deviation","humidity_deviation","disease_risk","smart_suggestion"],
        "🐛 Suitability (5)"  :["temp_in_range","humidity_in_range","ctrl_temp_in_range","ctrl_hum_in_range","silkworm_suitable"],
        "🌿 Derived Env (5)"  :["dew_point","heat_index","wind_chill","temp_humidity_index","temp_diff_apparent"],
        "📉 ML Lags (8)"      :["temp_lag1","temp_lag2","temp_lag3","humidity_lag1","humidity_lag2","humidity_lag3","wind_lag1","pres_lag1"],
        "📊 ML Rolling (14)"  :["temp_roll3_mean","temp_roll3_std","temp_roll6_mean","temp_roll6_std","temp_roll12_mean","temp_roll12_std","hum_roll3_mean","hum_roll3_std","hum_roll6_mean","hum_roll6_std","wind_roll6_mean","pres_roll6_mean"],
        "🎯 ML Targets (4)"   :["next_temp","next_humidity","next_action","next_action_label"],
    }
    gcols = st.columns(4)
    for i,(grp,cols) in enumerate(col_groups.items()):
        with gcols[i%4]:
            avail=[c for c in cols if c in df_full.columns]
            st.markdown(f"<div style='font-size:.8rem;font-weight:700;color:#64b5f6;margin:6px 0 4px;'>{grp}</div>",unsafe_allow_html=True)
            st.markdown(" ".join(f'<span class="mlchip">{c}</span>' for c in avail),unsafe_allow_html=True)

    st.divider()
    a1,a2 = st.columns(2)
    with a1:
        st.markdown("<div class='tsec-title'>🌡 Temperature & Humidity Summary</div>", unsafe_allow_html=True)
        num_df = df_full[["temperature","humidity","apparent_temp","dew_point","heat_index","wind_chill"]].describe().round(3)
        st.dataframe(num_df, use_container_width=True)
    with a2:
        st.markdown("<div class='tsec-title'>🌬 Wind, Pressure & Scores</div>", unsafe_allow_html=True)
        num_df2 = df_full[["wind_speed","pressure","visibility","health_score","temp_deviation","humidity_deviation"]].describe().round(3)
        st.dataframe(num_df2, use_container_width=True)

    st.divider()
    b1,b2,b3,b4 = st.columns(4)
    with b1:
        st.markdown("<div class='tsec-title'>🏷 Action Labels</div>", unsafe_allow_html=True)
        ac_df=df_full["action_label"].value_counts().reset_index(); ac_df.columns=["Action","Count"]
        ac_df["%"]=(ac_df["Count"]/ac_df["Count"].sum()*100).round(2).astype(str)+"%"
        st.dataframe(ac_df,use_container_width=True,hide_index=True)
    with b2:
        st.markdown("<div class='tsec-title'>🦠 Disease Risk</div>", unsafe_allow_html=True)
        dr_df=df_full["disease_risk"].value_counts().reset_index(); dr_df.columns=["Risk","Count"]
        dr_df["%"]=(dr_df["Count"]/dr_df["Count"].sum()*100).round(1).astype(str)+"%"
        st.dataframe(dr_df,use_container_width=True,hide_index=True)
    with b3:
        st.markdown("<div class='tsec-title'>🧠 Intel Status</div>", unsafe_allow_html=True)
        is_df=df_full["intel_status"].value_counts().reset_index(); is_df.columns=["Status","Count"]
        is_df["%"]=(is_df["Count"]/is_df["Count"].sum()*100).round(2).astype(str)+"%"
        st.dataframe(is_df,use_container_width=True,hide_index=True)
    with b4:
        st.markdown("<div class='tsec-title'>🌿 Season</div>", unsafe_allow_html=True)
        se_df=df_full["season"].value_counts().reset_index(); se_df.columns=["Season","Count"]
        se_df["%"]=(se_df["Count"]/se_df["Count"].sum()*100).round(1).astype(str)+"%"
        st.dataframe(se_df,use_container_width=True,hide_index=True)

    ldf2 = get_ldf()
    if ldf2 is not None:
        st.divider()
        sec("📋",f"Live Simulation Data — {len(ldf2)} steps","#64b5f6")
        show_cols=["timestamp","temperature","humidity","apparent_temp","wind_speed","pressure",
                   "dew_point","heat_index","action","action_label","controlled_temp",
                   "controlled_humidity","intel_status","health_score","disease_risk","season","precip_type"]
        show_cols=[c for c in show_cols if c in ldf2.columns]
        st.dataframe(ldf2[show_cols][::-1].reset_index(drop=True),use_container_width=True)
    else:
        st.markdown("<div class='ibox'>▶ Run the simulation to see live captured data here.</div>",unsafe_allow_html=True)

with t3:
    sec("📈","Full Visualizations","#64b5f6")

    monthly = df_full.groupby("month_name").agg(
        avg_temp=("temperature","mean"), avg_hum=("humidity","mean"),
        avg_ct=("controlled_temp","mean"), avg_ch=("controlled_humidity","mean"),
    ).reindex(["January","February","March","April","May","June",
               "July","August","September","October","November","December"]).dropna()

    with plt.rc_context(DARK_PLOT):
        fig,axes=plt.subplots(1,2,figsize=(13,4))
        x=range(len(monthly))
        axes[0].bar(x,monthly["avg_temp"],color="#ff6b6b",alpha=0.75,label="Raw",width=0.4,align="edge")
        axes[0].bar([i+0.4 for i in x],monthly["avg_ct"],color="#64ffda",alpha=0.75,label="Controlled",width=0.4,align="edge")
        axes[0].axhspan(TEMP_MIN,TEMP_MAX,color="#64ffda",alpha=0.06)
        axes[0].set_xticks(list(x)); axes[0].set_xticklabels(monthly.index,rotation=45,ha="right",fontsize=8)
        axes[0].set_title("Monthly Avg Temperature",fontweight="bold"); axes[0].legend(fontsize=8)
        axes[1].bar(x,monthly["avg_hum"],color="#64b5f6",alpha=0.75,label="Raw",width=0.4,align="edge")
        axes[1].bar([i+0.4 for i in x],monthly["avg_ch"],color="#64ffda",alpha=0.75,label="Controlled",width=0.4,align="edge")
        axes[1].axhspan(HUM_MIN,HUM_MAX,color="#64ffda",alpha=0.06)
        axes[1].set_xticks(list(x)); axes[1].set_xticklabels(monthly.index,rotation=45,ha="right",fontsize=8)
        axes[1].set_title("Monthly Avg Humidity",fontweight="bold"); axes[1].legend(fontsize=8)
        plt.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close(fig)

    hourly=df_full.groupby("hour").agg(avg_temp=("temperature","mean"),avg_hum=("humidity","mean"))
    with plt.rc_context(DARK_PLOT):
        fig,(a1,a2)=plt.subplots(1,2,figsize=(13,3.5))
        a1.fill_between(hourly.index,hourly["avg_temp"],alpha=0.2,color="#ff6b6b")
        a1.plot(hourly.index,hourly["avg_temp"],color="#ff6b6b",lw=1.8)
        a1.axhspan(TEMP_MIN,TEMP_MAX,color="#64ffda",alpha=0.06)
        a1.set_title("Avg Temperature by Hour",fontweight="bold"); a1.set_xlabel("Hour"); a1.set_ylabel("°C")
        a2.fill_between(hourly.index,hourly["avg_hum"],alpha=0.2,color="#64b5f6")
        a2.plot(hourly.index,hourly["avg_hum"],color="#64b5f6",lw=1.8)
        a2.axhspan(HUM_MIN,HUM_MAX,color="#64ffda",alpha=0.06)
        a2.set_title("Avg Humidity by Hour",fontweight="bold"); a2.set_xlabel("Hour"); a2.set_ylabel("%")
        plt.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close(fig)

    with plt.rc_context(DARK_PLOT):
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        axes[0].hist(df_full["health_score"].dropna(),bins=30,color="#8e44ad",alpha=0.85,edgecolor="#1a1f30")
        axes[0].set_title("Health Score Distribution",fontweight="bold"); axes[0].set_xlabel("Score (0–100)")
        sc=df_full["intel_status"].value_counts()
        cols_p=[{"Optimal":"#64ffda","Warning":"#ffd54f","Critical":"#ff6b6b"}.get(k,"#5a6a8a") for k in sc.index]
        wedges,texts,autotexts=axes[1].pie(sc.values,labels=sc.index,autopct="%1.1f%%",
                                            colors=cols_p,startangle=140,textprops={"fontsize":9})
        for at in autotexts: at.set_color("#1a1f30"); at.set_fontweight("bold")
        axes[1].set_title("Intel Status Distribution",fontweight="bold")
        dr=df_full["disease_risk"].value_counts()
        bar_colors=["#ff6b6b","#ffd54f","#ffb74d","#64ffda","#64b5f6"]
        axes[2].barh(dr.index,dr.values,color=bar_colors[:len(dr)],edgecolor="#1a1f30")
        axes[2].set_title("Disease Risk Counts",fontweight="bold"); axes[2].set_xlabel("Count")
        plt.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close(fig)

    seasons_order=["Spring","Summer","Autumn","Winter"]
    sdata=[df_full[df_full["season"]==s]["temperature"].dropna().values for s in seasons_order if s in df_full["season"].values]
    slbls=[s for s in seasons_order if s in df_full["season"].values]
    with plt.rc_context(DARK_PLOT):
        fig,(a1,a2)=plt.subplots(1,2,figsize=(13,4))
        bp=a1.boxplot(sdata,labels=slbls,patch_artist=True,
                      boxprops=dict(facecolor="#1e3a28",color="#64ffda"),
                      medianprops=dict(color="#ff6b6b",lw=2.5),
                      whiskerprops=dict(color="#3a6050"),capprops=dict(color="#3a6050"),
                      flierprops=dict(markerfacecolor="#ff6b6b",markersize=3,alpha=0.5))
        a1.axhspan(TEMP_MIN,TEMP_MAX,color="#64ffda",alpha=0.06)
        a1.set_title("Temperature by Season",fontweight="bold"); a1.set_ylabel("°C")
        wc_=df_full["wind_category"].value_counts()
        wcolors=["#2a3a5a","#3a5a8a","#4a7abf","#64b5f6"]
        a2.bar(wc_.index,wc_.values,color=wcolors[:len(wc_)],edgecolor="#1a1f30")
        a2.set_title("Wind Category Distribution",fontweight="bold"); a2.set_ylabel("Count")
        plt.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close(fig)

    ldf3=get_ldf()
    if ldf3 is not None:
        st.divider(); sec("🔴",f"Live Simulation Charts — {len(ldf3)} steps","#ff6b6b")
        xs3=list(range(len(ldf3)))
        with plt.rc_context(DARK_PLOT):
            fig,axes=plt.subplots(2,3,figsize=(14,7))
            axes[0,0].plot(xs3,ldf3["temperature"],color="#ff6b6b",lw=0.9,label="Raw")
            axes[0,0].plot(xs3,ldf3["controlled_temp"],color="#64ffda",lw=1.2,label="Ctrl")
            axes[0,0].axhspan(TEMP_MIN,TEMP_MAX,color="#64ffda",alpha=0.06)
            axes[0,0].set_title("Temperature",fontweight="bold"); axes[0,0].legend(fontsize=8)
            axes[0,1].plot(xs3,ldf3["humidity"],color="#64b5f6",lw=0.9,label="Raw")
            axes[0,1].plot(xs3,ldf3["controlled_humidity"],color="#64ffda",lw=1.2,label="Ctrl")
            axes[0,1].axhspan(HUM_MIN,HUM_MAX,color="#64ffda",alpha=0.06)
            axes[0,1].set_title("Humidity",fontweight="bold"); axes[0,1].legend(fontsize=8)
            axes[0,2].fill_between(xs3,ldf3["health_score"],alpha=0.2,color="#8e44ad")
            axes[0,2].plot(xs3,ldf3["health_score"],color="#ce93d8",lw=1.2)
            axes[0,2].axhline(70,color="#ffd54f",ls="--",lw=1); axes[0,2].axhline(40,color="#ff6b6b",ls="--",lw=1)
            axes[0,2].set_ylim(0,105); axes[0,2].set_title("Health Score",fontweight="bold")
            axes[1,0].plot(xs3,ldf3["wind_speed"],color="#ce93d8",lw=0.9); axes[1,0].set_title("Wind Speed",fontweight="bold")
            axes[1,1].plot(xs3,ldf3["pressure"],color="#64b5f6",lw=0.9); axes[1,1].set_title("Pressure (mb)",fontweight="bold")
            axes[1,2].plot(xs3,ldf3["dew_point"],color="#64ffda",lw=0.9,label="Dew Pt")
            axes[1,2].plot(xs3,ldf3["heat_index"],color="#ffb74d",lw=0.9,label="Heat Idx")
            axes[1,2].legend(fontsize=8); axes[1,2].set_title("Dew Point vs Heat Index",fontweight="bold")
            for ax in axes.flat: ax.set_xlabel("Step")
            plt.suptitle("Live Simulation — All Environmental Variables",fontweight="bold",fontsize=11,color="#c8cfe0")
            plt.tight_layout(pad=1.5)
        st.pyplot(fig); plt.close(fig)
    else:
        st.markdown("<div class='ibox'>▶ Run simulation to see live charts here.</div>",unsafe_allow_html=True)

with t4:
    sec("⚙️","Control System Results","#ffd54f")

    st.markdown("#### 📊 Control Performance — Full 96,429 Rows")
    st.markdown("<div class='kpi-strip'>"+
        "".join(f"<div class='kpi'><div class='kv {cls}'>{v}</div><div class='kl'>{l}</div></div>"
                for v,l,cls in [
            (f"{df_full['temp_in_range'].mean()*100:.1f}%","Temp In-Range (Raw)","cb"),
            (f"{df_full['ctrl_temp_in_range'].mean()*100:.1f}%","Temp In-Range (Ctrl)","cg"),
            (f"{df_full['humidity_in_range'].mean()*100:.1f}%","Hum In-Range (Raw)","cb"),
            (f"{df_full['ctrl_hum_in_range'].mean()*100:.1f}%","Hum In-Range (Ctrl)","cg"),
            (f"{df_full['temp_correction_delta'].mean():+.3f}°C","Avg Temp Correction","cy"),
            (f"{df_full['silkworm_suitable'].sum():,}","Suitable Steps","cg"),
        ])+"</div>", unsafe_allow_html=True)

    with plt.rc_context(DARK_PLOT):
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,4))
        cats=["Temp\nRaw","Temp\nCtrl","Hum\nRaw","Hum\nCtrl"]
        vals=[df_full["temp_in_range"].mean()*100,df_full["ctrl_temp_in_range"].mean()*100,
              df_full["humidity_in_range"].mean()*100,df_full["ctrl_hum_in_range"].mean()*100]
        bar_c=["#ff6b6b","#64ffda","#64b5f6","#64ffda"]
        bars=ax1.bar(cats,vals,color=bar_c,width=0.5,edgecolor="#1a1f30")
        for b,v in zip(bars,vals):
            ax1.text(b.get_x()+b.get_width()/2,b.get_height()+0.5,f"{v:.1f}%",
                     ha="center",va="bottom",fontweight="bold",fontsize=10,color="#e2e8f0")
        ax1.set_ylim(0,115); ax1.set_title("% Time In Optimal Range",fontweight="bold")
        tc=df_full["temp_correction_delta"].dropna()
        ax2.hist(tc,bins=50,color="#ffd54f",alpha=0.8,edgecolor="#1a1f30")
        ax2.axvline(0,color="#ff6b6b",lw=1.5,ls="--")
        ax2.set_title("Temperature Correction Δ Distribution",fontweight="bold")
        ax2.set_xlabel("Δ°C"); ax2.set_ylabel("Count")
        plt.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close(fig)

    ac_=df_full["action_label"].value_counts()
    with plt.rc_context(DARK_PLOT):
        fig,ax=plt.subplots(figsize=(8,4))
        bar_c2=["#ff6b6b","#ffb74d","#64b5f6","#64ffda","#ce93d8","#ffd54f"]
        bars=ax.bar(ac_.index,ac_.values,color=bar_c2[:len(ac_)],edgecolor="#1a1f30")
        for i,(lbl,v) in enumerate(ac_.items()):
            ax.text(i,v+300,f"{v:,}",ha="center",va="bottom",fontsize=9,fontweight="bold",color="#e2e8f0")
        ax.set_title("Action Label Distribution — Full Dataset",fontweight="bold"); ax.set_ylabel("Count")
        plt.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close(fig)

    ldf4=get_ldf()
    if ldf4 is not None:
        st.divider(); sec("🔴",f"Live Control — {len(ldf4)} steps","#ff6b6b")
        l1,l2,l3,l4=st.columns(4)
        l1.metric("Temp In-Range Raw",f"{(ldf4['temperature'].between(TEMP_MIN,TEMP_MAX)).mean()*100:.1f}%")
        l2.metric("Temp In-Range Ctrl",f"{(ldf4['controlled_temp'].between(TEMP_MIN,TEMP_MAX)).mean()*100:.1f}%")
        l3.metric("Hum In-Range Raw",f"{(ldf4['humidity'].between(HUM_MIN,HUM_MAX)).mean()*100:.1f}%")
        l4.metric("Hum In-Range Ctrl",f"{(ldf4['controlled_humidity'].between(HUM_MIN,HUM_MAX)).mean()*100:.1f}%")
        dc4=["timestamp","temperature","controlled_temp","temp_correction_delta",
             "humidity","controlled_humidity","hum_correction_delta",
             "action","action_label","intel_status","health_score","disease_risk"]
        dc4=[c for c in dc4 if c in ldf4.columns]
        st.dataframe(ldf4[dc4][::-1].reset_index(drop=True),use_container_width=True)
        st.download_button("⬇️ Download Control Results CSV",
            ldf4[dc4].to_csv(index=False).encode(),file_name="control_results.csv",mime="text/csv")
    else:
        st.markdown("<div class='ibox'>▶ Run simulation to see live control results.</div>",unsafe_allow_html=True)


with t5:
    sec("🧠","AI Intelligence Layer","#ce93d8")

    tot5=len(df_full)
    n_opt=(df_full["intel_status"]=="Optimal").sum()
    n_wrn=(df_full["intel_status"]=="Warning").sum()
    n_crt=(df_full["intel_status"]=="Critical").sum()
    avg_hs=df_full["health_score"].mean()
    flach=(df_full["disease_risk"].str.contains("Flacherie",na=False)).sum()
    grass=(df_full["disease_risk"].str.contains("Grasserie",na=False)).sum()
    musc =(df_full["disease_risk"].str.contains("Muscardine",na=False)).sum()
    safe =(df_full["disease_risk"]=="Safe").sum()

    st.markdown("<div class='kpi-strip'>"+
        "".join(f"<div class='kpi'><div class='kv {cl}'>{v}</div><div class='kl'>{l}</div><div class='ks {cl}'>{s}</div></div>"
                for v,l,s,cl in [
            (f"{n_opt:,}","🟢 Optimal",f"{n_opt/tot5*100:.2f}%","cg"),
            (f"{n_wrn:,}","🟡 Warning",f"{n_wrn/tot5*100:.2f}%","cy"),
            (f"{n_crt:,}","🔴 Critical",f"{n_crt/tot5*100:.2f}%","cr"),
            (f"{avg_hs:.1f}","💯 Avg Health Score","/ 100","cp"),
            (f"{flach:,}","⚠ Flacherie",f"{flach/tot5*100:.1f}%","cr"),
            (f"{grass:,}","⚠ Grasserie",f"{grass/tot5*100:.1f}%","cy"),
            (f"{musc:,}","⚠ Muscardine",f"{musc/tot5*100:.1f}%","cr"),
            (f"{safe:,}","✅ Safe",f"{safe/tot5*100:.1f}%","cg"),
        ])+"</div>", unsafe_allow_html=True)

    st.divider()
    sample_hs=df_full[["timestamp","health_score","intel_status"]].iloc[::50].reset_index(drop=True)
    with plt.rc_context(DARK_PLOT):
        fig,ax=plt.subplots(figsize=(14,3.5))
        cmap_={"Optimal":"#64ffda","Warning":"#ffd54f","Critical":"#ff6b6b"}
        for s_,c_ in cmap_.items():
            mask=sample_hs["intel_status"]==s_
            ax.scatter(sample_hs.index[mask],sample_hs["health_score"][mask],
                       color=c_,s=5,alpha=0.6,label=s_)
        ax.axhline(70,color="#ffd54f",ls="--",lw=1,alpha=0.7)
        ax.axhline(40,color="#ff6b6b",ls="--",lw=1,alpha=0.7)
        ax.fill_between(range(len(sample_hs)),70,105,alpha=0.04,color="#64ffda")
        ax.fill_between(range(len(sample_hs)),40,70,alpha=0.04,color="#ffd54f")
        ax.fill_between(range(len(sample_hs)),0,40,alpha=0.04,color="#ff6b6b")
        ax.set_title("Health Score — Full Dataset (every 50th row)",fontweight="bold")
        ax.set_ylabel("Score (0–100)"); ax.legend(fontsize=9,ncol=3); ax.set_ylim(0,108)
        plt.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close(fig)

    disease_heat=df_full.copy()
    disease_heat["has_risk"]=(disease_heat["disease_risk"]!="Safe").astype(int)
    pivot=disease_heat.pivot_table("has_risk","hour","month",aggfunc="mean")
    with plt.rc_context(DARK_PLOT):
        fig,ax=plt.subplots(figsize=(10,4))
        im=ax.imshow(pivot.values,aspect="auto",cmap="RdYlGn_r",vmin=0,vmax=1,interpolation="nearest")
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Month"); ax.set_ylabel("Hour of Day")
        ax.set_title("Disease Risk Rate — Hour × Month",fontweight="bold")
        cbar=plt.colorbar(im,ax=ax,label="Risk Rate"); cbar.ax.yaxis.label.set_color("#c8cfe0")
        plt.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close(fig)

    ldf5=get_ldf()
    if ldf5 is not None:
        st.divider(); sec("🔴",f"Live Intelligence — {len(ldf5)} steps","#ff6b6b")
        l1,l2,l3,l4=st.columns(4)
        l1.metric("🟢 Optimal",f"{(ldf5['intel_status']=='Optimal').sum()}")
        l2.metric("🟡 Warning",f"{(ldf5['intel_status']=='Warning').sum()}")
        l3.metric("🔴 Critical",f"{(ldf5['intel_status']=='Critical').sum()}")
        l4.metric("Avg Health",f"{ldf5['health_score'].mean():.1f}/100")
        last5=ldf5.iloc[-1]; s_l,sc_l,_,_=status_info(last5["temperature"],last5["humidity"])
        sc_l_={"Optimal":"#64ffda","Warning":"#ffd54f","Critical":"#ff6b6b"}[s_l]
        cl5,cr5=st.columns([1,2])
        with cl5:
            st.markdown(f"""
            <div class='scard {"green" if s_l=="Optimal" else ("yellow" if s_l=="Warning" else "red")}'>
            <div class='slbl'>LATEST STEP SNAPSHOT</div>
            <div style='font-size:.9rem;color:#e2e8f0;line-height:2;'>
            <b>Temp:</b> {last5['temperature']:.2f}°C &nbsp;|&nbsp; <b>Humidity:</b> {last5['humidity']:.1f}%<br>
            <b>Status:</b> <span style='color:{sc_l_};font-weight:700;'>{s_l}</span> &nbsp;|&nbsp; <b>Score:</b> <span style='color:{sc_l_};'>{sc_l:.1f}/100</span><br>
            <b>Disease:</b> {last5['disease_risk']}<br>
            <b>Season:</b> {last5.get('season','')} &nbsp;|&nbsp; <b>Hour:</b> {last5.get('hour','')}
            </div></div>
            """, unsafe_allow_html=True)
        with cr5:
            alert5=(f"<div class='critbox'>🚨 UNSAFE FOR SILKWORMS!</div>" if s_l=="Critical" else "")
            st.markdown(alert5+intel_panel_html(last5["temperature"],last5["humidity"],
                [r["temperature"] for r in st.session_state["rows"]]),unsafe_allow_html=True)
        with plt.rc_context(DARK_PLOT):
            fig,(ia1,ia2)=plt.subplots(1,2,figsize=(13,3.5))
            xs5=list(range(len(ldf5)))
            ia1.fill_between(xs5,ldf5["health_score"],alpha=0.2,color="#8e44ad")
            ia1.plot(xs5,ldf5["health_score"],color="#ce93d8",lw=1.3)
            ia1.axhline(70,color="#ffd54f",ls="--",lw=1); ia1.axhline(40,color="#ff6b6b",ls="--",lw=1)
            ia1.fill_between(xs5,[70]*len(xs5),ldf5["health_score"].clip(upper=100),
                             where=ldf5["health_score"]>=70,alpha=0.06,color="#64ffda")
            ia1.set_ylim(0,108); ia1.set_title("Live Health Score",fontweight="bold"); ia1.set_xlabel("Step")
            sc5v=ldf5["intel_status"].value_counts()
            cols5=[{"Optimal":"#64ffda","Warning":"#ffd54f","Critical":"#ff6b6b"}.get(k,"#5a6a8a") for k in sc5v.index]
            wedges,texts,autotexts=ia2.pie(sc5v.values,labels=sc5v.index,autopct="%1.1f%%",colors=cols5,startangle=140)
            for at in autotexts: at.set_color("#1a1f30"); at.set_fontweight("bold")
            ia2.set_title("Status Distribution",fontweight="bold")
            plt.tight_layout(pad=1.5)
        st.pyplot(fig); plt.close(fig)
    else:
        st.markdown("<div class='ibox'>▶ Run simulation to see live intelligence panels.</div>",unsafe_allow_html=True)


with t6:
    sec("🤖","ML Prediction Layer","#ce93d8")

    if not SKLEARN_OK:
        st.error("❌ scikit-learn not installed — run: pip install scikit-learn"); st.stop()
    if ml is None:
        st.error("❌ ML training failed. Check dataset."); st.stop()

    st.markdown(f"""<div class='mlbox'>
    <b>Trained on {ml['n_train']:,} rows</b> (time-based 80/20 split — no data leakage) ·
    <b>{len(ml['feat_cols'])} features</b> (lags, rolling stats, derived env, health score, time encoding) ·
    <b>RF:</b> 100 estimators, max_depth=12 · <b>Ridge:</b> α=1.0
    </div>""", unsafe_allow_html=True)

    mc1,mc2,mc3=st.columns(3)
    acc_c="cg" if ml["acc"]>0.9 else ("cy" if ml["acc"]>0.75 else "cr")
    mc1.markdown(
        f"<div class='scard purple mlcard'><div class='slbl' style='color:#ce93d8;'>🌲 RF CLASSIFIER — ACTION</div>"
        f"<div class='sval {acc_c}'>{ml['acc']*100:.2f}%</div>"
        f"<div class='ssub' style='color:#8060a0;'>Accuracy · {ml['n_test']:,} test rows</div>"
        f"<div class='ssub' style='color:#7050a0;margin-top:4px;'>Classes: {', '.join(ml['classes'])}</div></div>",
        unsafe_allow_html=True)
    t_c="cg" if ml["t_r2"]>0.9 else ("cy" if ml["t_r2"]>0.75 else "cr")
    mc2.markdown(
        f"<div class='scard purple mlcard'><div class='slbl' style='color:#ce93d8;'>🌡 RF REGRESSOR — TEMP</div>"
        f"<div class='sval {t_c}'>R² {ml['t_r2']:.4f}</div>"
        f"<div class='ssub' style='color:#8060a0;'>MAE {ml['t_mae']:.4f}°C · RMSE {ml['t_rmse']:.4f}°C</div></div>",
        unsafe_allow_html=True)
    h_c="cg" if ml["h_r2"]>0.9 else ("cy" if ml["h_r2"]>0.75 else "cr")
    mc3.markdown(
        f"<div class='scard purple mlcard'><div class='slbl' style='color:#ce93d8;'>💧 RIDGE — HUMIDITY</div>"
        f"<div class='sval {h_c}'>R² {ml['h_r2']:.4f}</div>"
        f"<div class='ssub' style='color:#8060a0;'>MAE {ml['h_mae']:.4f}% · RMSE {ml['h_rmse']:.4f}%</div></div>",
        unsafe_allow_html=True)

    st.divider()
    fi1,fi2=st.columns(2)
    with fi1:
        st.markdown("<div style='font-size:.82rem;font-weight:700;color:#ce93d8;margin-bottom:8px;'>🔍 Feature Importances — Classifier (Top 15)</div>", unsafe_allow_html=True)
        imp_c=pd.Series(ml["imp_clf"],index=ml["feat_cols"]).nlargest(15).sort_values()
        with plt.rc_context(DARK_PLOT):
            fig,ax=plt.subplots(figsize=(6,5))
            colors_fi=[f"#{int(100+i*10):02x}{int(50+i*5):02x}{int(180+i*4):02x}" for i in range(len(imp_c))]
            ax.barh(imp_c.index,imp_c.values,color="#8e44ad",alpha=0.85,edgecolor="#1a1f30")
            ax.set_title("RF Classifier Feature Importances",fontweight="bold")
            ax.set_xlabel("Importance"); plt.tight_layout(pad=1.5)
        st.pyplot(fig); plt.close(fig)
    with fi2:
        st.markdown("<div style='font-size:.82rem;font-weight:700;color:#64b5f6;margin-bottom:8px;'>🔍 Feature Importances — Temp Regressor (Top 15)</div>", unsafe_allow_html=True)
        imp_r=pd.Series(ml["imp_rfr"],index=ml["feat_cols"]).nlargest(15).sort_values()
        with plt.rc_context(DARK_PLOT):
            fig,ax=plt.subplots(figsize=(6,5))
            ax.barh(imp_r.index,imp_r.values,color="#2980b9",alpha=0.85,edgecolor="#1a1f30")
            ax.set_title("RF Regressor Feature Importances",fontweight="bold")
            ax.set_xlabel("Importance"); plt.tight_layout(pad=1.5)
        st.pyplot(fig); plt.close(fig)

    st.divider()
    ap1,ap2=st.columns(2)
    with ap1:
        with plt.rc_context(DARK_PLOT):
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            axes[0].scatter(ml["ytte"],ml["yt_pred"],alpha=0.12,s=4,color="#ff6b6b")
            mn_=min(ml["ytte"].min(),ml["yt_pred"].min()); mx_=max(ml["ytte"].max(),ml["yt_pred"].max())
            axes[0].plot([mn_,mx_],[mn_,mx_],color="#64ffda",lw=1.2,ls="--",label="Perfect")
            axes[0].set_title("Temp: Actual vs Predicted",fontweight="bold")
            axes[0].set_xlabel("Actual (°C)"); axes[0].set_ylabel("Predicted (°C)"); axes[0].legend(fontsize=8)
            resid_t=ml["ytte"]-ml["yt_pred"]
            axes[1].hist(resid_t,bins=60,color="#ff6b6b",alpha=0.8,edgecolor="#1a1f30")
            axes[1].axvline(0,color="#64ffda",lw=1.5,ls="--")
            axes[1].set_title("Temp Residuals",fontweight="bold"); axes[1].set_xlabel("Residual (°C)")
            plt.tight_layout(pad=1.5)
        st.pyplot(fig); plt.close(fig)
    with ap2:
        with plt.rc_context(DARK_PLOT):
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            axes[0].scatter(ml["yhte"],ml["yh_pred"],alpha=0.12,s=4,color="#64b5f6")
            mn_=min(ml["yhte"].min(),ml["yh_pred"].min()); mx_=max(ml["yhte"].max(),ml["yh_pred"].max())
            axes[0].plot([mn_,mx_],[mn_,mx_],color="#64ffda",lw=1.2,ls="--",label="Perfect")
            axes[0].set_title("Humidity: Actual vs Predicted",fontweight="bold")
            axes[0].set_xlabel("Actual (%)"); axes[0].set_ylabel("Predicted (%)"); axes[0].legend(fontsize=8)
            resid_h=ml["yhte"]-ml["yh_pred"]
            axes[1].hist(resid_h,bins=60,color="#64b5f6",alpha=0.8,edgecolor="#1a1f30")
            axes[1].axvline(0,color="#64ffda",lw=1.5,ls="--")
            axes[1].set_title("Humidity Residuals",fontweight="bold"); axes[1].set_xlabel("Residual (%)")
            plt.tight_layout(pad=1.5)
        st.pyplot(fig); plt.close(fig)

    with plt.rc_context(DARK_PLOT):
        fig,ax=plt.subplots(figsize=(8,5))
        cm_=ml["cm"]
        im=ax.imshow(cm_,interpolation="nearest",cmap="Blues")
        plt.colorbar(im,ax=ax)
        for i in range(cm_.shape[0]):
            for j in range(cm_.shape[1]):
                ax.text(j,i,str(cm_[i,j]),ha="center",va="center",fontsize=8,
                        color="white" if cm_[i,j]>cm_.max()/2 else "#c8cfe0",fontweight="bold")
        ax.set_xticks(range(len(ml["classes"]))); ax.set_xticklabels(ml["classes"],rotation=30,ha="right",fontsize=9)
        ax.set_yticks(range(len(ml["classes"]))); ax.set_yticklabels(ml["classes"],fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix — Action Classifier",fontweight="bold")
        plt.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close(fig)

    st.divider()
    sec("🔴","Live ML Inference & Forecast","#ce93d8")
    ldf6=get_ldf()
    if ldf6 is None or len(ldf6)<4:
        st.markdown("<div class='ibox'>▶ Run simulation (≥4 steps) to see live inference and forecast.</div>",unsafe_allow_html=True)
    else:
        last_rec=ldf6.iloc[-1].to_dict()
        pa_,pt_,ph_=live_infer(ml,last_rec)
        if pt_ is not None:
            lt,lh=ldf6["temperature"].iloc[-1],ldf6["humidity"].iloc[-1]
            li1,li2,li3=st.columns(3)
            li1.markdown(mlcard("🤖 PREDICTED NEXT ACTION",pa_,"RF Classifier · 100 trees"),unsafe_allow_html=True)
            li2.markdown(mlcard("🌡 PREDICTED NEXT TEMP",f"{pt_:.2f}°C",f"Actual: {lt:.2f}°C · Δ {pt_-lt:+.2f}°C"),unsafe_allow_html=True)
            li3.markdown(mlcard("💧 PREDICTED NEXT HUMIDITY",f"{ph_:.1f}%",f"Actual: {lh:.1f}% · Δ {ph_-lh:+.2f}%"),unsafe_allow_html=True)
            _,sc_p,_,_=status_info(pt_,ph_)
            dr_p=disease_risk_str(pt_,ph_); sg_p=suggestions_str(pt_,ph_)
            suit_p=TEMP_MIN<=pt_<=TEMP_MAX and HUM_MIN<=ph_<=HUM_MAX
            st.markdown(f"""<div class='mlbox'>
            <b>🔮 Predicted Next-Step Assessment:</b> &nbsp;
            Score: <b>{sc_p}/100</b> &nbsp;|&nbsp; Disease: <b>{dr_p}</b> &nbsp;|&nbsp;
            Silkworm Suitable: <b>{"✅ YES" if suit_p else "❌ NO"}</b> &nbsp;|&nbsp;
            Action: <b>{sg_p}</b>
            </div>""",unsafe_allow_html=True)

        st.markdown(f"<div style='font-size:.9rem;font-weight:700;color:#ce93d8;margin:16px 0 8px;'>🔮 {ml_horizon}-Step Ahead Forecast</div>",unsafe_allow_html=True)
        t_fc,h_fc=rolling_forecast(ml,ldf6,horizon=ml_horizon)
        if t_fc:
            n6=len(ldf6)
            past_x=list(range(n6)); fut_x=list(range(n6-1,n6+len(t_fc)))
            t_br=[ldf6["temperature"].iloc[-1]]+t_fc
            h_br=[ldf6["humidity"].iloc[-1]]+h_fc
            with plt.rc_context(DARK_PLOT):
                fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,4.5))
                ax1.plot(past_x,ldf6["temperature"],color="#ff6b6b",lw=0.9,label="Historical",alpha=0.9)
                ax1.plot(fut_x,t_br,color="#ce93d8",lw=2.2,ls="--",marker="o",markersize=4,label=f"Forecast ({ml_horizon} steps)")
                ax1.fill_between(fut_x,[v-ml["t_mae"] for v in t_br],[v+ml["t_mae"] for v in t_br],
                                 alpha=0.2,color="#8e44ad",label=f"±MAE ({ml['t_mae']:.2f}°C)")
                ax1.axhspan(TEMP_MIN,TEMP_MAX,color="#64ffda",alpha=0.06,label="Optimal Range")
                ax1.axvline(n6-1,color="#5a6a8a",ls=":",lw=1.2,alpha=0.7)
                ax1.set_title("Temperature Forecast",fontweight="bold"); ax1.set_xlabel("Step"); ax1.set_ylabel("°C"); ax1.legend(fontsize=8,ncol=2)
                ax2.plot(past_x,ldf6["humidity"],color="#64b5f6",lw=0.9,label="Historical",alpha=0.9)
                ax2.plot(fut_x,h_br,color="#ce93d8",lw=2.2,ls="--",marker="o",markersize=4,label=f"Forecast ({ml_horizon} steps)")
                ax2.fill_between(fut_x,[v-ml["h_mae"] for v in h_br],[v+ml["h_mae"] for v in h_br],
                                 alpha=0.2,color="#8e44ad",label=f"±MAE ({ml['h_mae']:.2f}%)")
                ax2.axhspan(HUM_MIN,HUM_MAX,color="#64ffda",alpha=0.06,label="Optimal Range")
                ax2.axvline(n6-1,color="#5a6a8a",ls=":",lw=1.2,alpha=0.7)
                ax2.set_title("Humidity Forecast",fontweight="bold"); ax2.set_xlabel("Step"); ax2.set_ylabel("%"); ax2.legend(fontsize=8,ncol=2)
                plt.suptitle(f"🔮 {ml_horizon}-Step ML Forecast with ±MAE Confidence Band",fontweight="bold",fontsize=11,color="#c8cfe0")
                plt.tight_layout(pad=1.5)
            st.pyplot(fig); plt.close(fig)

            fc_df=pd.DataFrame({
                "Step":range(1,len(t_fc)+1),
                "Pred Temp (°C)":t_fc,"Pred Hum (%)":h_fc,
                "Temp OK":["✅" if TEMP_MIN<=v<=TEMP_MAX else "❌" for v in t_fc],
                "Hum OK":["✅" if HUM_MIN<=v<=HUM_MAX else "❌" for v in h_fc],
                "Suitable":["✅" if TEMP_MIN<=t_<=TEMP_MAX and HUM_MIN<=h_<=HUM_MAX else "❌" for t_,h_ in zip(t_fc,h_fc)],
                "Pred Action":[determine_action(t_,h_) for t_,h_ in zip(t_fc,h_fc)],
                "Status":[status_info(t_,h_)[0] for t_,h_ in zip(t_fc,h_fc)],
                "Health":[status_info(t_,h_)[1] for t_,h_ in zip(t_fc,h_fc)],
                "Disease":[disease_risk_str(t_,h_) for t_,h_ in zip(t_fc,h_fc)],
            })
            st.dataframe(fc_df,use_container_width=True,hide_index=True)
            st.download_button("⬇️ Download Forecast CSV",fc_df.to_csv(index=False).encode(),file_name="forecast.csv",mime="text/csv")

        st.divider()
        ml_pts=[(p,a) for p,a in zip(st.session_state["ml_pt"][:-1],ldf6["temperature"].tolist()[1:]) if p is not None]
        ml_phs=[(p,a) for p,a in zip(st.session_state["ml_ph"][:-1],ldf6["humidity"].tolist()[1:])    if p is not None]
        if len(ml_pts)>=3:
            prt=[x[0] for x in ml_pts]; art=[x[1] for x in ml_pts]
            prh=[x[0] for x in ml_phs]; arh=[x[1] for x in ml_phs]
            la1,la2,la3,la4=st.columns(4)
            la1.metric("Live Temp MAE",f"{mean_absolute_error(art,prt):.3f}°C")
            la2.metric("Live Temp RMSE",f"{np.sqrt(mean_squared_error(art,prt)):.3f}°C")
            la3.metric("Live Hum MAE",f"{mean_absolute_error(arh,prh):.3f}%")
            la4.metric("Live Hum RMSE",f"{np.sqrt(mean_squared_error(arh,prh)):.3f}%")
            with plt.rc_context(DARK_PLOT):
                fig,(lax1,lax2)=plt.subplots(1,2,figsize=(14,3.5))
                xsv=list(range(len(art)))
                lax1.plot(xsv,art,color="#ff6b6b",lw=1,label="Actual Temp")
                lax1.plot(xsv,prt,color="#ce93d8",lw=0.9,ls="--",label="ML Predicted")
                lax1.fill_between(xsv,[a_-ml["t_mae"] for a_ in prt],[a_+ml["t_mae"] for a_ in prt],alpha=0.12,color="#8e44ad")
                lax1.axhspan(TEMP_MIN,TEMP_MAX,color="#64ffda",alpha=0.06)
                lax1.set_title("Temp: Actual vs ML Predicted",fontweight="bold"); lax1.legend(fontsize=8)
                lax2.plot(xsv,arh,color="#64b5f6",lw=1,label="Actual Hum")
                lax2.plot(xsv,prh,color="#ce93d8",lw=0.9,ls="--",label="ML Predicted")
                lax2.fill_between(xsv,[a_-ml["h_mae"] for a_ in prh],[a_+ml["h_mae"] for a_ in prh],alpha=0.12,color="#8e44ad")
                lax2.axhspan(HUM_MIN,HUM_MAX,color="#64ffda",alpha=0.06)
                lax2.set_title("Humidity: Actual vs ML Predicted",fontweight="bold"); lax2.legend(fontsize=8)
                for ax in (lax1,lax2): ax.set_xlabel("Step")
                plt.tight_layout(pad=1.5)
            st.pyplot(fig); plt.close(fig)
        else:
            st.markdown("<div class='ibox'>Run ≥5 simulation steps for live accuracy tracking.</div>",unsafe_allow_html=True)

    st.divider()
    st.markdown("""<div class='mlbox'><b>📚 ML Features (38 pre-built from enhanced dataset):</b><br>"""+
        " ".join(f'<span class="mlchip">{c}</span>' for c in [
            "temperature","humidity","apparent_temp","wind_speed","pressure","visibility",
            "dew_point","heat_index","wind_chill","health_score","temp_deviation","humidity_deviation",
            "temp_lag1","temp_lag2","temp_lag3","humidity_lag1","humidity_lag2","humidity_lag3",
            "wind_lag1","pres_lag1","temp_roll3_mean","temp_roll3_std","hum_roll3_mean","hum_roll3_std",
            "hour_sin","hour_cos","month_sin","month_cos","is_daytime",
        ])+"</div>",unsafe_allow_html=True)

with t7:
    sec("📖","System Architecture & Documentation","#64b5f6")
    c1,c2=st.columns(2)
    with c1:
        st.markdown(f"""<div class='tsec'>
        <div class='tsec-title'>🎓 Project Info</div>
        <table style='width:100%;border-collapse:collapse;font-size:.85rem;color:#c8cfe0;'>
        <tr><td style='padding:5px 0;color:#5a6a8a;'>Title</td><td>Environmental Data Analysis for Automated T&H Control in Sericulture</td></tr>
        <tr><td style='padding:5px 0;color:#5a6a8a;'>Type</td><td>Simulation-Based IoT + ML</td></tr>
        <tr><td style='padding:5px 0;color:#5a6a8a;'>Institution</td><td>VIT Vellore</td></tr>
        <tr><td style='padding:5px 0;color:#5a6a8a;'>Semester</td><td>Winter 2025–26</td></tr>
        <tr><td style='padding:5px 0;color:#5a6a8a;'>Dataset</td><td>weatherHistory_enhanced.csv</td></tr>
        <tr><td style='padding:5px 0;color:#5a6a8a;'>Rows</td><td>{len(df_full):,}</td></tr>
        <tr><td style='padding:5px 0;color:#5a6a8a;'>Columns</td><td>76 (12 original + 64 engineered)</td></tr>
        <tr><td style='padding:5px 0;color:#5a6a8a;'>Years</td><td>{df_full['year'].min()} – {df_full['year'].max()}</td></tr>
        </table></div>""",unsafe_allow_html=True)
    with c2:
        acc_show = f"{ml['acc']*100:.2f}%" if ml else "N/A"
        tr2_show = f"{ml['t_r2']:.4f}" if ml else "N/A"
        hr2_show = f"{ml['h_r2']:.4f}" if ml else "N/A"
        st.markdown(f"""<div class='tsec'>
        <div class='tsec-title'>🤖 ML Models</div>
        <table style='width:100%;border-collapse:collapse;font-size:.85rem;color:#c8cfe0;'>
        <tr style='color:#5a6a8a;font-size:.76rem;'><td>Model</td><td>Algorithm</td><td>Metric</td></tr>
        <tr><td style='padding:6px 0;'>Action Classifier</td><td>Random Forest 100 trees</td><td style='color:#64ffda;font-weight:700;'>{acc_show}</td></tr>
        <tr><td style='padding:6px 0;'>Temp Regressor</td><td>Random Forest 100 trees</td><td style='color:#64ffda;font-weight:700;'>R² {tr2_show}</td></tr>
        <tr><td style='padding:6px 0;'>Humidity Regressor</td><td>Ridge Regression α=1</td><td style='color:#64ffda;font-weight:700;'>R² {hr2_show}</td></tr>
        </table></div>""",unsafe_allow_html=True)

    st.markdown(f"""<div class='tsec' style='margin-top:12px;'>
    <div class='tsec-title'>🛠 Technology Stack & Control Logic</div>
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;font-size:.85rem;color:#c8cfe0;'>
    <div>
    <b style='color:#64b5f6;'>Stack</b><br><br>
    Python 3.10+ · Streamlit · scikit-learn<br>
    Pandas · NumPy · Matplotlib
    </div>
    <div>
    <b style='color:#64b5f6;'>Control Logic</b><br><br>
    Temp &gt; {TEMP_MAX}°C → <span style='color:#ffd54f;'>Fan ON</span><br>
    Temp &lt; {TEMP_MIN}°C → <span style='color:#ffb74d;'>Heater ON</span><br>
    Hum &lt; {HUM_MIN}% → <span style='color:#64ffda;'>Humidifier ON</span><br>
    Hum &gt; {HUM_MAX}% → <span style='color:#64b5f6;'>Dehumidifier ON</span><br>
    Multiple → <span style='color:#ff6b6b;'>Multi-Action</span>
    </div>
    </div></div>""",unsafe_allow_html=True)

    st.markdown("<div style='text-align:center;margin-top:20px;font-size:.75rem;color:#2a3a5a;'>VIT Vellore · Winter Semester 2025–26 · weatherHistory_enhanced.csv · 96,429 rows · 76 features</div>",unsafe_allow_html=True)
