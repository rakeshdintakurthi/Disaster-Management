"""
ResQVision — Streamlit Command-Center Dashboard
=================================================
Professional disaster-response UI with live feed,
risk indicators, metric cards, charts, detection logs,
action panel, and dispatch simulation.

Run:  streamlit run ui_dashboard.py
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import json
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from datetime import datetime
from collections import deque

# ── Ensure local imports work ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import ResQVisionPipeline


# ═══════════════════════════════════════════════════════════════════════
# Page config & custom CSS
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ResQVision — Disaster Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* ── Government Command-Center Theme ───────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-primary:   #EEF2F6; /* Slightly cooler blue-gray background */
    --bg-card:      #FFFFFF;
    --bg-card-alt:  #F8FAFC;
    --accent-blue:  #3B82F6; /* Brighter, more engaging blue */
    --accent-red:   #EF4444; /* Brighter red */
    --accent-yellow:#F59E0B; /* Warm engaging amber */
    --accent-green: #10B981; /* Emerald green */
    --text-primary: #0F172A;
    --text-muted:   #475569;
    --border-color: #E2E8F0;
    --shadow-soft:  0 4px 15px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.03);
}

.stApp {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* Hide streamlit defaults */
#MainMenu, footer, header { visibility: hidden; }

/* ── Typography & Containers ──────────────────────────────────── */
h1, h2, h3, h4 {
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    margin-bottom: 0.5rem !important;
}

hr {
    border-color: var(--border-color);
    margin: 24px 0;
}

/* ── Metric cards ─────────────────────────────────────────────── */
.metric-card {
    background-color: #FFFFFF;
    border: 1px solid var(--border-color);
    border-radius: 8px; /* Sharper */
    padding: 16px 20px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05); /* very sharp, minimal shadow */
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 100%; height: 4px;
    background-color: var(--accent-blue);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-muted);
}
.metric-sublabel {
    font-size: 0.75rem;
    color: var(--text-muted);
    opacity: 0.8;
}

/* ── Alerts ─────────────────────────────────────────────── */
.alert-card {
    background-color: var(--bg-card);
    border-left: 4px solid var(--accent-red);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 24px;
    box-shadow: var(--shadow-soft);
    display: flex;
    align-items: center;
    gap: 16px;
}
.alert-card-warning {
    border-left: 4px solid var(--accent-yellow);
}
.alert-card-safe {
    border-left: 4px solid var(--accent-green);
}
.alert-icon {
    font-size: 1.5rem;
}
.alert-content h4 {
    margin: 0;
    font-size: 1rem;
    color: var(--text-primary);
}
.alert-content p {
    margin: 0;
    font-size: 0.9rem;
    color: var(--text-muted);
}

.pulse-dot {
    height: 12px;
    width: 12px;
    background-color: var(--accent-red);
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.7);
    animation: pulse-red-dot 2s infinite;
}

@keyframes pulse-red-dot {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
}

/* ── Panel card ───────────────────────────────────────────────── */
.panel {
    background-color: #FFFFFF;
    border: 1px solid var(--border-color);
    border-radius: 8px; /* Sharper */
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05); /* very sharp, minimal shadow */
}
.panel-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 8px;
}

/* ── Video Feed Container ────────────────────────────────────── */
.video-container {
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: var(--shadow-soft);
    margin-bottom: 16px;
}
.video-header {
    background-color: var(--bg-primary);
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.85rem;
    color: var(--text-muted);
    font-weight: 500;
}

/* ── Report box ───────────────────────────────────────────────── */
.report-box {
    background-color: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 16px;
    font-family: inherit;
    font-size: 0.85rem;
    color: var(--text-primary);
    white-space: pre-wrap;
    max-height: 400px;
    overflow-y: auto;
}

/* ── Subheader titles ────────────────────────────────────────── */
.section-divider {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-top: 32px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-color);
/* ── Primary Action Buttons ────────────────────────────────────────── */
button[kind="primary"] {
    background: linear-gradient(180deg, #059669 0%, #047857 100%) !important;
    border: 1px solid #065F46 !important;
    border-bottom: 4px solid #022C22 !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
    padding-top: 10px !important;
    padding-bottom: 10px !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
    transition: all 0.1s ease-in-out !important;
}

button[kind="primary"]:hover {
    background: linear-gradient(180deg, #10B981 0%, #059669 100%) !important;
    color: white !important;
    transform: translateY(-1px);
}

button[kind="primary"]:active {
    background: linear-gradient(180deg, #047857 0%, #064E3B 100%) !important;
    border-bottom: 0px solid #022C22 !important;
    transform: translateY(4px);
    margin-bottom: 4px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Session state initialisation
# ═══════════════════════════════════════════════════════════════════════

def init_state():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "running" not in st.session_state:
        st.session_state.running = False
    if "source" not in st.session_state:
        st.session_state.source = "0"
    if "history" not in st.session_state:
        st.session_state.history = {
            "confidence": deque(maxlen=200),
            "risk_score": deque(maxlen=200),
            "motion": deque(maxlen=200),
            "survivors": deque(maxlen=200),
            "timestamps": deque(maxlen=200),
        }
    if "last_report" not in st.session_state:
        st.session_state.last_report = ""
    if "dispatch_count" not in st.session_state:
        st.session_state.dispatch_count = 0
    if "paused" not in st.session_state:
        st.session_state.paused = False
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

init_state()


# ═══════════════════════════════════════════════════════════════════════
# Helper: build Plotly gauge
# ═══════════════════════════════════════════════════════════════════════

def make_gauge(value: float, title: str, max_val: float = 1.0,
               suffix: str = "%", color: str = "#2563EB") -> go.Figure:
    display_val = value * 100 if max_val <= 1.0 else value
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_val,
        number={"suffix": suffix, "font": {"size": 24, "color": "#0F172A"}},
        title={"text": title, "font": {"size": 14, "color": "#475569"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#E2E8F0", "showticklabels": False},
            "bar": {"color": color, "thickness": 0.15},
            "bgcolor": "#F1F5F9",
            "borderwidth": 0,
        },
    ))
    fig.update_layout(
        height=150, margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e7ef",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Helper: metric card HTML
# ═══════════════════════════════════════════════════════════════════════

def metric_card(label: str, value, color: str = "#2563EB", sublabel: str = "") -> str:
    sub = f'<div class="metric-sublabel">{sublabel}</div>' if sublabel else ""
    return f"""
    <div class="metric-card" style="border-left: 4px solid {color};">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        {sub}
    </div>
    """


# (Sidebar removed to simplify UI)
st.sidebar.markdown("## ⚙️ System Status\\n_All controls have been moved to the main dashboard for easier access._")


# ═══════════════════════════════════════════════════════════════════════
# Handle Start / Stop
# ═══════════════════════════════════════════════════════════════════════

if 'start_btn' in locals() and start_btn:
    if source is None:
        st.error("Please upload a file first.")
    else:
        st.session_state.pipeline = ResQVisionPipeline(mode=mode)
        st.session_state.pipeline.detector.confidence_threshold = conf_threshold
        if gemini_api_key:
            st.session_state.pipeline.gemini_agent.update_key(gemini_api_key)
        st.session_state.running = True

if 'stop_btn' in locals() and stop_btn:
    st.session_state.running = False


# ═══════════════════════════════════════════════════════════════════════
# Main layout
# ═══════════════════════════════════════════════════════════════════════

# ── Header ───────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 24px; border-radius: 8px; background-color: #1E40AF; display: flex; align-items: center; gap: 20px; margin-bottom: 30px;">
    <div style="font-size: 2.5rem; background-color: #3B82F6; padding: 12px; border-radius: 8px;">🛡️</div>
    <div>
        <h1 style="color: #FFFFFF; margin: 0; font-size: 1.8rem; font-weight: 700; letter-spacing: 0.5px;">ResQVision Command Center</h1>
        <div style="color: #E2E8F0; font-size: 1rem; font-weight: 500; letter-spacing: 0.5px;">Advanced Disaster Intelligence & Real-time Telemetry</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Main Controls (Moved from sidebar) ───────────────────────────────
if not st.session_state.running:
    st.markdown('<div class="panel"><div class="panel-title">⚙️ Configuration & Launch</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    with c1:
        mode = st.selectbox("Scenario Mode", ["flood", "rubble"], index=0)
        conf_threshold = st.slider("Detection Confidence", 0.05, 0.9, 0.2, 0.05)
        gemini_api_key = st.text_input("Gemini API Key (Optional)", type="password", help="Enables advanced AI situational awareness.")
        
    with c2:
        input_type = st.selectbox("Input Type", ["File Upload", "Webcam / URL"], index=0)
        source = "0"
        if input_type == "Webcam / URL":
            source = st.text_input("Video Source", value="0", help="0 = webcam, or path to video file")
        else:
            uploaded_file = st.file_uploader("Upload Video/Photo", type=["mp4", "avi", "mov", "jpg", "jpeg", "png"])
            if uploaded_file is not None:
                upload_dir = os.path.join("logs", "uploads")
                os.makedirs(upload_dir, exist_ok=True)
                source = os.path.abspath(os.path.join(upload_dir, uploaded_file.name))
                with open(source, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"File ready: {uploaded_file.name}")
            else:
                source = None
                
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        start_btn = st.button("▶ START ANALYSIS", use_container_width=True, type="primary")
        if start_btn:
            if source is None:
                st.error("Please upload a file first.")
            else:
                st.session_state.source = source
                st.session_state.pipeline = ResQVisionPipeline(mode=mode)
                st.session_state.pipeline.detector.confidence_threshold = conf_threshold
                if gemini_api_key:
                    st.session_state.pipeline.gemini_agent.update_key(gemini_api_key)
                st.session_state.running = True
                st.session_state.paused = False
                st.session_state.last_result = None
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
else:
    mode = st.session_state.pipeline.mode
    source = st.session_state.source
    c_btn1, c_btn2, c_btn3 = st.columns(3)
    with c_btn1:
        if st.button("■ STOP ANALYSIS", type="primary", use_container_width=True):
            st.session_state.running = False
            st.session_state.paused = False
            st.session_state.last_result = None
            if st.session_state.get("cap"):
                st.session_state.cap.release()
                st.session_state.cap = None
            st.rerun()
    with c_btn2:
        if not st.session_state.paused:
            if st.button("⏸ PAUSE FEED", use_container_width=True):
                st.session_state.paused = True
                st.rerun()
        else:
            if st.button("▶ RESUME FEED", type="primary", use_container_width=True):
                st.session_state.paused = False
                st.rerun()
    with c_btn3:
        if st.button("🔄 RESTART FEED", use_container_width=True):
            if st.session_state.get("cap"):
                st.session_state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            st.session_state.paused = False
            st.rerun()

# ── Process one frame ────────────────────────────────────────────────
if st.session_state.running and st.session_state.pipeline is not None:
    src = int(source) if source.isdigit() else source
    is_photo = (source is not None and str(source).lower().endswith(('.jpg', '.jpeg', '.png')))

    if is_photo:
        frame = cv2.imread(str(src))
        if frame is not None:
            result = st.session_state.pipeline.run_pipeline_frame(frame, is_photo=True)
            st.session_state.last_result = result

            h = st.session_state.history
            h["confidence"].append(result["detection"]["confidence_score"])
            h["risk_score"].append(result["risk"]["risk_score"])
            h["motion"].append(result["motion"]["micro_motion_confidence"])
            h["survivors"].append(result["detection"]["survivor_count"])
            h["timestamps"].append(datetime.now().strftime("%H:%M:%S"))

            if result["report"]:
                st.session_state.last_report = result["report"]
                
            st.session_state.running = False
            st.info("📷 Photo analysis complete.")
        else:
            st.error(f"❌ Cannot open image source: {src}")
            st.session_state.running = False
            
    else:
        if "cap" not in st.session_state or st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(src)

        cap = st.session_state.cap
        if cap.isOpened() and not st.session_state.paused:
            ret, frame = cap.read()
            if ret:
                result = st.session_state.pipeline.run_pipeline_frame(frame, is_photo=False)
                
                st.session_state.last_result = result

                h = st.session_state.history
                h["confidence"].append(result["detection"]["confidence_score"])
                h["risk_score"].append(result["risk"]["risk_score"])
                h["motion"].append(result["motion"]["micro_motion_confidence"])
                h["survivors"].append(result["detection"]["survivor_count"])
                h["timestamps"].append(datetime.now().strftime("%H:%M:%S"))

                if result["report"]:
                    st.session_state.last_report = result["report"]
            else:
                st.session_state.paused = True
                cap.release()
                st.session_state.cap = None
                st.warning("⚠ Video stream ended. (Dashboard Paused)")
        elif not cap.isOpened():
            st.error(f"❌ Cannot open video source: {src}")
            st.session_state.running = False

# ═══════════════════════════════════════════════════════════════════════
# Dashboard panels
# ═══════════════════════════════════════════════════════════════════════

result = st.session_state.get("last_result")
if result is not None:
    risk_level = result["risk"]["risk_level"]
    risk_score = result["risk"]["risk_score"]
    
    # ── Override risk based on Gemini ────────────────────────────────
    gemini = result.get("gemini_analysis", {})
    if gemini.get("status") == "success" and gemini.get("overriding_risk_score") is not None:
        risk_score = gemini["overriding_risk_score"]
        # Fast re-bucket
        if risk_score >= 80: risk_level = "CRITICAL"
        elif risk_score >= 55: risk_level = "HIGH"
        elif risk_score >= 30: risk_level = "MEDIUM"
        else: risk_level = "LOW"

    # ── Alerts ────────────────────────────────────────
    if risk_level == "CRITICAL":
        alert_class = "alert-card"
        icon = '<span class="pulse-dot"></span>'
        title = "SYSTEM STATUS: CRITICAL ALERT"
        msg = "Immediate rescue operations required. High probability of casualties."
    elif risk_level == "HIGH":
        alert_class = "alert-card alert-card-warning"
        icon = '⚠️'
        title = "SYSTEM STATUS: HIGH RISK"
        msg = "Elevated danger detected. Prepare for possible deployment."
    else:
        alert_class = "alert-card alert-card-safe"
        icon = '✅'
        title = "SYSTEM STATUS: STABLE"
        msg = "Monitoring active. No immediate critical threats detected."
        
    st.markdown(f"""
        <div class="{alert_class}">
            <div class="alert-icon">{icon}</div>
            <div class="alert-content">
                <h4>{title}</h4>
                <p>{msg}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider">Situation Overview</div>', unsafe_allow_html=True)

    # ── Top metric cards ─────────────────────────────────────────────
    mc = st.columns(5)
    risk_colors = {"CRITICAL": "#DC2626", "HIGH": "#F59E0B",
                   "MEDIUM": "#F59E0B", "LOW": "#16A34A"}
    rc = risk_colors.get(risk_level, "#2563EB")

    with mc[0]:
        st.markdown(metric_card("Survivors Confirmed",
                                result["detection"]["survivor_count"], "#2563EB", sublabel="Detected via visual feed"),
                    unsafe_allow_html=True)
    with mc[1]:
        st.markdown(metric_card("Risk Level", risk_level, rc, sublabel="Aggregate threat status"),
                    unsafe_allow_html=True)
    with mc[2]:
        st.markdown(metric_card("Risk Score",
                                f"{risk_score}/100", rc, sublabel="Weighted composite score"),
                    unsafe_allow_html=True)
    with mc[3]:
        st.markdown(metric_card("AI Confidence",
                                f"{result['detection']['confidence_score']:.0%}",
                                "#2563EB", sublabel="Detection certainty"),
                    unsafe_allow_html=True)
    with mc[4]:
        st.markdown(metric_card("System Health", f"{result['fps']} FPS", "#16A34A", sublabel="Processing speed"),
                    unsafe_allow_html=True)

    st.markdown("")

    # ── Gemini SITREP Panel ──────────────────────────────────────────
    if gemini.get("status") == "success":
        st.markdown('<div class="section-divider">Intelligence & Recommendations</div>', unsafe_allow_html=True)
        st.markdown(f'''
        <div class="panel" style="border-left: 4px solid #2563EB;">
            <div class="panel-title"><span>🤖 AI Command SITREP</span> <span style="font-weight: normal; font-size: 0.8rem;">Gemini 2.5 Flash</span></div>
            <p style="color: #0F172A; font-size: 1rem; line-height: 1.5; margin-bottom: 12px;">{gemini.get('sitrep_summary', '')}</p>
            <div style="background-color: rgba(37, 99, 235, 0.05); padding: 12px; border-radius: 6px; border: 1px solid rgba(37, 99, 235, 0.15);">
                <strong style="color: #2563EB;">Tactical Advice:</strong> <span style="color: #0F172A;">{gemini.get('tactical_advice', '')}</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('<div class="section-divider">Live Analysis</div>', unsafe_allow_html=True)
    vid_col, gauge_col = st.columns([3, 1])

    with vid_col:
        st.markdown(f'''
        <div class="video-container">
            <div class="video-header">
                <span>🔴 LIVE FEED</span>
                <span>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>
            </div>
        ''', unsafe_allow_html=True)
        tab_det, tab_heat = st.tabs(["Vision Output", "Thermal Density"])
        with tab_det:
            rgb = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
            st.image(rgb, channels="RGB", use_column_width=False, width=800)
        with tab_heat:
            rgb_h = cv2.cvtColor(result["heatmap_frame"], cv2.COLOR_BGR2RGB)
            st.image(rgb_h, channels="RGB", use_column_width=False, width=800)
        st.markdown('</div>', unsafe_allow_html=True)

    with gauge_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_gauge(result["detection"]["confidence_score"],
                       "Detection Confidence", color="#2563EB"),
            use_container_width=True, key="gauge_conf",
        )
        st.markdown('<hr style="margin:0;">', unsafe_allow_html=True)
        st.plotly_chart(
            make_gauge(result["motion"]["micro_motion_confidence"],
                       "Micro-Motion", color="#2563EB"),
            use_container_width=True, key="gauge_motion",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Timeline charts ──────────────────────────────────────────────
    st.markdown('<div class="section-divider">Telemetry Logs</div>', unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)
    h = st.session_state.history

    with ch1:
        if len(h["confidence"]) > 1:
            df_conf = pd.DataFrame({
                "Time": list(h["timestamps"]),
                "Confidence": list(h["confidence"]),
                "Risk Score": [v / 100 for v in h["risk_score"]],
            })
            st.line_chart(df_conf.set_index("Time"), height=220)

    with ch2:
        if len(h["motion"]) > 1:
            df_mot = pd.DataFrame({
                "Time": list(h["timestamps"]),
                "Micro-Motion": list(h["motion"]),
                "Survivors": [s / max(max(h["survivors"]), 1) for s in h["survivors"]],
            })
            st.line_chart(df_mot.set_index("Time"), height=220)

    # ── Bottom panels: Strategy / Resources ─────────────────────────
    p1, p2 = st.columns(2)

    with p1:
        strat = result["strategy"]
        equip_html = "".join([f'<li style="margin-bottom: 4px; color: #0F172A;">{eq}</li>' for eq in strat["equipment"][:5]])
        actions_html = "".join([f'<li style="margin-bottom: 4px; color: #0F172A;">{act}</li>' for act in strat["actions"]])
        
        st.markdown(f'''
        <div class="panel" style="border-top: 4px solid #3B82F6; background-color: #FFFFFF; height: 100%;">
            <div class="panel-title" style="color: #2563EB;">🎯 Strategy</div>
            <p style="color: #0F172A;"><strong>{strat['priority']}</strong></p>
            <p style="color: #475569; font-style: italic; margin-bottom: 12px;">{strat['strategy']}</p>
            <p style="color: #0F172A; margin-bottom: 4px;"><strong>Equipment:</strong></p>
            <ul style="margin-top: 0; padding-left: 20px; font-size: 0.9rem;">{equip_html}</ul>
            <details style="margin-top: 12px;">
                <summary style="cursor: pointer; font-weight: 600; color: #2563EB; font-size: 0.95rem;">Actions Deployment</summary>
                <ul style="margin-top: 8px; padding-left: 20px; font-size: 0.9rem;">{actions_html}</ul>
            </details>
        </div>
        ''', unsafe_allow_html=True)

    with p2:
        res = result["resources"]
        st.markdown(f'''
        <div class="panel" style="border-top: 4px solid #10B981; background-color: #FFFFFF; height: 100%;">
            <div class="panel-title" style="color: #059669;">🚑 Resources</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 10px;">
                <div>
                    <div style="font-size: 0.85rem; color: #475569; font-weight: 500; text-transform: uppercase;">Ambulances</div>
                    <div style="font-size: 1.8rem; color: #0F172A; font-weight: 700;">{res["ambulances"]}</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: #475569; font-weight: 500; text-transform: uppercase;">Helicopters</div>
                    <div style="font-size: 1.8rem; color: #0F172A; font-weight: 700;">{res["helicopters"]}</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: #475569; font-weight: 500; text-transform: uppercase;">Rescue Teams</div>
                    <div style="font-size: 1.8rem; color: #0F172A; font-weight: 700;">{res["rescue_teams"]}</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: #475569; font-weight: 500; text-transform: uppercase;">Supply Trucks</div>
                    <div style="font-size: 1.8rem; color: #0F172A; font-weight: 700;">{res["supply_trucks"]}</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: #475569; font-weight: 500; text-transform: uppercase;">Medical Units</div>
                    <div style="font-size: 1.8rem; color: #0F172A; font-weight: 700;">{res["medical_units"]}</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: #475569; font-weight: 500; text-transform: uppercase;">Police Units</div>
                    <div style="font-size: 1.8rem; color: #0F172A; font-weight: 700;">{res["police_units"]}</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: #475569; font-weight: 500; text-transform: uppercase;">Crowd Staff</div>
                    <div style="font-size: 1.8rem; color: #0F172A; font-weight: 700;">{res["crowd_control_staff"]}</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: #2563EB; font-weight: 600; text-transform: uppercase;">Total Personnel</div>
                    <div style="font-size: 1.8rem; color: #1E3A8A; font-weight: 800;">{res["total_personnel"]}</div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # ── Map & Report Container ─────────────────────────────────────────
    st.markdown('<div class="section-divider">Incident Mapping & Reporting</div>', unsafe_allow_html=True)
    r1, r2 = st.columns([1, 1])

    with r1:
        report = st.session_state.last_report or "_Generating first report…_"
        st.markdown(f'''
        <div class="panel" style="border: 2px solid #F59E0B; background-color: #FFFFFF; border-radius: 8px; height: 100%;">
            <div class="panel-title" style="color: #D97706; border-bottom: 1px solid #E2E8F0;">📋 Incident Report</div>
            <div class="report-box" style="border-color: rgba(245, 158, 11, 0.3); background-color: #FFFFFF; color: #452E04; max-height: 380px;">{report}</div>
        ''', unsafe_allow_html=True)
        
        if report and "Generating" not in report:
            st.download_button(
                label="📥 Export Report (TXT)",
                data=report,
                file_name=f"Incident_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with r2:
        st.markdown(f'''
        <div class="panel" style="border-top: 4px solid #8B5CF6; background-color: #FFFFFF; height: 100%;">
            <div class="panel-title" style="color: #7C3AED;">🗺️ Live Deployment Map</div>
        ''', unsafe_allow_html=True)
        
        # Determine Coordinates from Telemetry (if available) or default to a dummy location
        telemetry = result.get("drone_telemetry", {})
        gps = telemetry.get("gps", [19.0760, 72.8777]) # default to Mumbai
        lat, lon = gps[0], gps[1]
        
        # Create Folium Map
        m = folium.Map(location=[lat, lon], zoom_start=15, tiles='cartodbpositron')
        
        # Add Incident Marker
        folium.Marker(
            [lat, lon],
            popup="<i>Incident Center</i>",
            tooltip="Incident Epicenter",
            icon=folium.Icon(color="red", icon="warning-sign")
        ).add_to(m)
        
        # Draw a deployment radius
        folium.Circle(
            radius=300,
            location=[lat, lon],
            popup="Deployment Drop Zone",
            color="#8B5CF6",
            fill=True,
            fill_color="#C4B5FD",
            fill_opacity=0.3
        ).add_to(m)
        
        # Render Map in Streamlit
        st_folium(m, height=413, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Detection logs table ─────────────────────────────────────────
    csv_path = os.path.join("logs", "detections.csv")
    if os.path.exists(csv_path):
        with st.expander("📄 Detection Logs"):
            df_log = pd.read_csv(csv_path)
            st.dataframe(df_log.tail(50), use_container_width=True, height=250)

    # ── Dispatch Rescue button ───────────────────────────────────────
    st.markdown("---")
    dcol1, dcol2, dcol3 = st.columns([2, 1, 2])
    with dcol2:
        if st.button("🚁 DISPATCH RESCUE", type="primary", use_container_width=True):
            st.session_state.dispatch_count += 1
            st.balloons()
            res = result['resources']
            st.success(f"""
**✅ Rescue Team #{st.session_state.dispatch_count} Dispatched Successfully!**

The alert and deployment payload was sent to the following units:
- **Ambulances**: {res['ambulances']}
- **Helicopters**: {res['helicopters']}
- **Rescue Teams**: {res['rescue_teams']}
- **Supply Trucks**: {res['supply_trucks']}
- **Medical Units**: {res['medical_units']}
- **Personnel**: {res['total_personnel']}

*Estimated Time of Arrival (ETA): 15 minutes.*
            """)

    # ── Auto-rerun for live feed ─────────────────────────────────────
    if st.session_state.running and not st.session_state.paused:
        time.sleep(0.03)
        st.rerun()

else:
    # ── Idle state ───────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:80px 20px;">
        <div style="font-size:4rem; margin-bottom:16px;">🛡️</div>
        <div style="font-size:1.3rem; color:#7a8a9e; letter-spacing:2px;">
            SYSTEM STANDBY
        </div>
        <div style="font-size:0.85rem; color:#4a5568; margin-top:12px;">
            Configure settings in the sidebar and press <strong>▶ START</strong> to begin.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Demo mode with sample data ───────────────────────────────────
    with st.expander("🧪 Demo — Preview Dashboard with Sample Data"):
        demo_cols = st.columns(5)
        with demo_cols[0]:
            st.markdown(metric_card("Survivors", 4, "#00e5ff"), unsafe_allow_html=True)
        with demo_cols[1]:
            st.markdown(metric_card("Risk Level", "HIGH", "#ff6d00"), unsafe_allow_html=True)
        with demo_cols[2]:
            st.markdown(metric_card("Risk Score", "72/100", "#ff6d00"), unsafe_allow_html=True)
        with demo_cols[3]:
            st.markdown(metric_card("Confidence", "87%", "#76ff03"), unsafe_allow_html=True)
        with demo_cols[4]:
            st.markdown(metric_card("FPS", 24.5, "#ffc107"), unsafe_allow_html=True)

        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(make_gauge(0.87, "Detection Confidence"),
                           use_container_width=True)
        with g2:
            st.plotly_chart(make_gauge(0.42, "Micro-Motion", color="#76ff03"),
                           use_container_width=True)
