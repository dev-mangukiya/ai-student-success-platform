import sys
import os

# ==========================
# FIX PROJECT IMPORT PATH
# ==========================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import dill
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

from src.components.rag_advisor import RAGAdvisor
from src.config import REGRESSOR_MODEL_PATH, PREPROCESSOR_PATH

# ==========================
# RESPONSIVE PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Student Success AI | Telemetry",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================
# PREMIUM GLOBAL CSS INJECTION (MASSIVE UI OVERHAUL)
# ==========================
st.markdown(
    """
    <style>
        /* Import Premium Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');
        
        /* Global Reset & Base Styling */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
            background-color: #0B0F19 !important; /* Deep space background */
            color: #E2E8F0 !important;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif !important;
        }

        /* Top Bar Override */
        header[data-testid="stHeader"] {
            background: transparent !important;
        }

        /* Main Container */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 5rem !important;
        }

        /* Animations */
        @keyframes fadeUp {
            0% { opacity: 0; transform: translateY(30px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes glowPulse {
            0% { box-shadow: 0 0 15px rgba(56, 189, 248, 0.1); }
            50% { box-shadow: 0 0 30px rgba(56, 189, 248, 0.4); }
            100% { box-shadow: 0 0 15px rgba(56, 189, 248, 0.1); }
        }

        .animate-in {
            animation: fadeUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        /* Glassmorphism Metric Cards */
        .glass-card {
            background: rgba(17, 24, 39, 0.65);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            padding: 32px;
            transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
            position: relative;
            overflow: hidden;
        }
        
        .glass-card:hover {
            transform: translateY(-8px);
            border-color: rgba(56, 189, 248, 0.4);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), inset 0 0 0 1px rgba(255, 255, 255, 0.1);
        }

        /* Glow effects for cards */
        .glass-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 100%;
            background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0) 100%);
            pointer-events: none;
        }

        /* Sidebar Styling Override */
        [data-testid="stSidebar"] {
            background: #0F172A !important;
            border-right: 1px solid rgba(255,255,255,0.05);
        }
        
        [data-testid="stSidebar"] * {
            font-family: 'Outfit', sans-serif !important;
        }

        /* Sidebar Input elements */
        div[data-baseweb="select"] > div {
            background-color: #1E293B !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            color: #F8FAFC !important;
        }
        .stSlider > div > div > div > div { background-color: #38BDF8 !important; }
        
        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 16px;
            background: rgba(15, 23, 42, 0.6);
            padding: 10px 10px 0 10px;
            border-radius: 16px 16px 0 0;
            border-bottom: 2px solid rgba(255,255,255,0.05);
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 10px 10px 0 0;
            padding: 12px 24px;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 600;
            font-size: 1.1rem;
            color: #64748B;
            border: none;
            transition: all 0.3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #F8FAFC;
        }
        .stTabs [aria-selected="true"] {
            color: #38BDF8 !important;
            background: rgba(56, 189, 248, 0.1) !important;
            border-bottom: 3px solid #38BDF8 !important;
        }

        /* AI Report Container */
        .ai-report-container {
            background: linear-gradient(145deg, #1E293B, #0F172A);
            padding: 40px 50px;
            border-radius: 24px;
            border: 1px solid rgba(56, 189, 248, 0.2);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            line-height: 1.8;
            font-size: 1.1rem;
            color: #F8FAFC;
            animation: glowPulse 4s infinite;
        }
        
        /* Button Styling */
        div[data-testid="stButton"] button {
            background: linear-gradient(135deg, #38BDF8, #818CF8) !important;
            border: none !important;
            border-radius: 12px !important;
            color: white !important;
            font-weight: 700 !important;
            font-family: 'Outfit', sans-serif !important;
            padding: 0.75rem 1.5rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 10px 20px rgba(56, 189, 248, 0.3) !important;
        }
        div[data-testid="stButton"] button:hover {
            transform: translateY(-2px) scale(1.02) !important;
            box-shadow: 0 15px 30px rgba(56, 189, 248, 0.5) !important;
        }
        
        /* Metric Label Styling */
        .metric-label {
            font-family: 'Outfit', sans-serif;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-size: 0.85rem;
            font-weight: 700;
            color: #94A3B8;
            margin-bottom: 12px;
        }
        
        .metric-value {
            font-family: 'Outfit', sans-serif;
            font-size: 4.5rem;
            font-weight: 800;
            line-height: 1;
            background: linear-gradient(to right, #F8FAFC, #94A3B8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .gradient-text {
            background: linear-gradient(135deg, #38BDF8, #818CF8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================
# INITIALIZE SESSION STATE
# ==========================
if "prediction_triggered" not in st.session_state:
    st.session_state.prediction_triggered = False
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "avg_score" not in st.session_state:
    st.session_state.avg_score = None
if "weak_features" not in st.session_state:
    st.session_state.weak_features = []
if "ai_advice" not in st.session_state:
    st.session_state.ai_advice = None
if "transformed_input" not in st.session_state:
    st.session_state.transformed_input = None


# ==========================
# HERO HEADER SECTION
# ==========================
st.markdown(
    """
    <div class='animate-in' style='padding: 30px 20px 40px 20px; text-align: center;'>
        <div style='display: inline-block; background: rgba(56, 189, 248, 0.1); padding: 8px 16px; border-radius: 30px; border: 1px solid rgba(56, 189, 248, 0.3); margin-bottom: 20px;'>
            <span style='color: #38BDF8; font-weight: 700; font-size: 0.9rem; letter-spacing: 1px;'>NEXT-GEN TELEMETRY</span>
        </div>
        <h1 style='margin: 0; font-size: 3.8rem; font-weight: 800; letter-spacing: -1px; color: #F8FAFC;'>Student Success <span class="gradient-text">Intelligence</span></h1>
        <p style='color: #94A3B8; margin: 15px auto 0 auto; font-size: 1.25rem; font-weight: 400; max-width: 600px; line-height: 1.6;'>
            Harnessing predictive neural networks and LLM advisory to decode and optimize academic trajectories.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================
# LOAD ARTIFACTS & RESOURCES
# ==========================
@st.cache_resource
def load_artifacts():
    with open(REGRESSOR_MODEL_PATH, "rb") as file:
        model = dill.load(file)

    with open(PREPROCESSOR_PATH, "rb") as file:
        preprocessor = dill.load(file)

    return model, preprocessor


@st.cache_resource
def load_advisor():
    return RAGAdvisor()


model, preprocessor = load_artifacts()
advisor = load_advisor()


# ==========================
# SIDEBAR CONTROLS
# ==========================
with st.sidebar:
    st.markdown(
        "<h2 style='color: #F8FAFC; font-weight: 800; font-size: 1.5rem; margin-bottom: 5px;'>System Config</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color: #64748B; font-size: 0.9rem; margin-bottom: 25px;'>Input student parameters to generate predictive telemetry.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h4 style='color: #94A3B8; font-weight: 600; font-size: 1rem; margin-top: 10px;'>Demographics</h4>",
        unsafe_allow_html=True,
    )
    gender = st.selectbox("Gender Identity", ["male", "female"])
    race = st.selectbox(
        "Demographic Group", ["group A", "group B", "group C", "group D", "group E"]
    )

    st.markdown(
        "<h4 style='color: #94A3B8; font-weight: 600; font-size: 1rem; margin-top: 20px;'>Background</h4>",
        unsafe_allow_html=True,
    )
    parental = st.selectbox(
        "Parental Education",
        [
            "some high school",
            "high school",
            "some college",
            "associate's degree",
            "bachelor's degree",
            "master's degree",
        ],
    )
    lunch = st.selectbox("Meal Plan", ["standard", "free/reduced"])
    test_preparation_course = st.selectbox(
        "Prep Course Completion", ["completed", "none"]
    )

    st.markdown(
        "<hr style='border: none; height: 1px; background: rgba(255,255,255,0.1); margin: 30px 0;'/>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h4 style='color: #94A3B8; font-weight: 600; font-size: 1rem;'>Current Telemetrics</h4>",
        unsafe_allow_html=True,
    )
    math_score = st.slider("📐 Quantitative Logic (Math)", 0, 100, 70)
    reading_score = st.slider("📖 Comprehension (Reading)", 0, 100, 70)
    writing_score = st.slider("✍️ Expression (Writing)", 0, 100, 70)

    st.markdown("<br>", unsafe_allow_html=True)

    predict_clicked = st.button(
        "Initialize Sequence 🚀", use_container_width=True, type="primary"
    )


# ==========================
# PREDICTION ENGINE
# ==========================
if predict_clicked:
    st.session_state.prediction_triggered = True

    input_data = pd.DataFrame(
        {
            "gender": [gender],
            "race/ethnicity": [race],
            "parental level of education": [parental],
            "lunch": [lunch],
            "test preparation course": [test_preparation_course],
            "math score": [math_score],
            "reading score": [reading_score],
            "writing score": [writing_score],
        }
    )

    # Cache transformed data for SHAP tab
    st.session_state.transformed_input = preprocessor.transform(input_data)

    pred = model.predict(st.session_state.transformed_input)[0]
    st.session_state.prediction = pred

    # Core Metric Risk Allocations
    weak = []
    if math_score < 60:
        weak.append("Mathematics Focus")
    if reading_score < 60:
        weak.append("Reading Comprehension")
    if writing_score < 60:
        weak.append("Written Synthesizing")
    if test_preparation_course == "none":
        weak.append("Preparatory Mock Absence")

    st.session_state.weak_features = weak
    st.session_state.avg_score = (math_score + reading_score + writing_score) / 3
    st.session_state.ai_advice = None


# ==========================
# DISPLAY INTERFACE
# ==========================
if st.session_state.prediction_triggered:
    dashboard, charts, explain, ai = st.tabs(
        [
            "⊞ Command Center",
            "🌌 Neural Visuals",
            "🧠 XAI Explainability",
            "🤖 Gemini Strategist",
        ]
    )

    # ==========================
    # TAB 1: EXECUTIVE DASHBOARD
    # ==========================
    with dashboard:
        st.markdown(
            "<div class='animate-in' style='margin-top: 25px;'>", unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown(
                f"""
                <div class='glass-card' style='display: flex; flex-direction: column; justify-content: center; align-items: center;'>
                    <div class='metric-label'>FORECAST HORIZON</div>
                    <div class='metric-value gradient-text'>{st.session_state.prediction:.1f}<span style='font-size: 2.5rem; opacity: 0.6;'>%</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class='glass-card' style='display: flex; flex-direction: column; justify-content: center; align-items: center;'>
                    <div class='metric-label'>CURRENT BASELINE</div>
                    <div class='metric-value'>{st.session_state.avg_score:.1f}<span style='font-size: 2.5rem; opacity: 0.6;'>%</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            risk_color = (
                "#F43F5E" if len(st.session_state.weak_features) > 0 else "#10B981"
            )
            risk_glow = (
                "rgba(244,63,94,0.3)"
                if len(st.session_state.weak_features) > 0
                else "rgba(16,185,129,0.3)"
            )
            st.markdown(
                f"""
                <div class='glass-card' style='box-shadow: 0 10px 30px {risk_glow}; border-color: rgba({int(risk_color[1:3], 16)}, {int(risk_color[3:5], 16)}, {int(risk_color[5:7], 16)}, 0.4); display: flex; flex-direction: column; justify-content: center; align-items: center;'>
                    <div class='metric-label' style='color: {risk_color};'>ACTIVE ANOMALIES</div>
                    <div class='metric-value' style='color: {risk_color}; background: none; -webkit-text-fill-color: {risk_color};'>{len(st.session_state.weak_features)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)

        if st.session_state.weak_features:
            weak_items_html = "".join(
                [
                    f"<div style='background: rgba(244,63,94,0.1); border-left: 4px solid #F43F5E; padding: 16px 24px; margin: 12px 0; border-radius: 8px; font-weight: 500; color: #FDA4AF; display: flex; align-items: center; gap: 12px; font-family: \"Outfit\", sans-serif;'><span style='font-size: 1.2rem;'>⚠️</span> {item}</div>"
                    for item in st.session_state.weak_features
                ]
            )
            st.markdown(
                f"""
                <div class='glass-card' style='border-top: 4px solid #E11D48;'>
                    <h3 style='margin: 0 0 20px 0; color: #F8FAFC; font-weight: 800; font-size: 1.6rem;'>System Alerts & Vulnerabilities</h3>
                    {weak_items_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class='glass-card' style='border-top: 4px solid #10B981; text-align: center; padding: 50px;'>
                    <div style='font-size: 3.5rem; margin-bottom: 20px;'>✨</div>
                    <h3 style='margin: 0; color: #F8FAFC; font-weight: 800; font-size: 1.8rem;'>Optimal Trajectory Confirmed</h3>
                    <p style='color: #94A3B8; margin: 10px 0 0 0; font-size: 1.15rem;'>All base metrics are perfectly stabilized. You are clear for high-altitude academic performance.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================
    # TAB 2: ADVANCED VISUALIZATIONS
    # ==========================
    with charts:
        st.markdown(
            "<div class='animate-in' style='margin-top: 25px;'>", unsafe_allow_html=True
        )

        score_df = pd.DataFrame(
            {
                "Subject": ["Math", "Reading", "Writing"],
                "Score": [math_score, reading_score, writing_score],
            }
        )

        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            bar = px.bar(
                score_df,
                x="Subject",
                y="Score",
                text="Score",
                title="Capability Matrix",
                color="Subject",
                color_discrete_map={
                    "Math": "#38BDF8",
                    "Reading": "#818CF8",
                    "Writing": "#C084FC",
                },
            )
            bar.update_traces(
                textposition="inside",
                textfont=dict(size=18, color="white", family="Outfit", weight="bold"),
                marker_line_width=0,
            )
            bar.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=60, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title_font=dict(size=22, color="#F8FAFC", family="Outfit"),
                font=dict(color="#94A3B8"),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
            )
            st.plotly_chart(bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with chart_col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=st.session_state.prediction,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={
                        "text": "Velocity Target",
                        "font": {"size": 22, "color": "#F8FAFC", "family": "Outfit"},
                    },
                    number={
                        "font": {"size": 60, "color": "#38BDF8", "family": "Outfit"}
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickwidth": 0,
                            "tickcolor": "white",
                        },
                        "bar": {"color": "#38BDF8", "thickness": 0.75},
                        "bgcolor": "rgba(255,255,255,0.05)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 55], "color": "rgba(244,63,94,0.4)"},
                            {"range": [55, 75], "color": "rgba(250,204,21,0.4)"},
                            {"range": [75, 100], "color": "rgba(16,185,129,0.4)"},
                        ],
                    },
                )
            )
            gauge.update_layout(
                margin=dict(l=20, r=20, t=70, b=10),
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(gauge, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

        radar_center_col, _ = st.columns([1.5, 1])
        with radar_center_col:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            radar = go.Figure()
            radar.add_trace(
                go.Scatterpolar(
                    r=[math_score, reading_score, writing_score, math_score],
                    theta=[
                        "Quantitative",
                        "Comprehension",
                        "Synthesizing",
                        "Quantitative",
                    ],
                    fill="toself",
                    fillcolor="rgba(56, 189, 248, 0.2)",
                    line=dict(color="#38BDF8", width=3, shape="spline"),
                    marker=dict(size=8, color="#818CF8"),
                )
            )
            radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        gridcolor="rgba(255,255,255,0.1)",
                        linecolor="rgba(0,0,0,0)",
                        tickfont=dict(color="#64748B"),
                    ),
                    angularaxis=dict(
                        gridcolor="rgba(255,255,255,0.1)",
                        linecolor="rgba(0,0,0,0)",
                        tickfont=dict(color="#94A3B8", size=14),
                    ),
                ),
                title="Symmetry & Alignment Profile",
                margin=dict(l=50, r=50, t=80, b=50),
                height=450,
                paper_bgcolor="rgba(0,0,0,0)",
                title_font=dict(size=22, color="#F8FAFC", family="Outfit"),
            )
            st.plotly_chart(radar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================
    # TAB 3: AI EXPLAINABILITY (SHAP)
    # ==========================
    with explain:
        st.markdown(
            "<div class='animate-in' style='margin-top: 25px;'>", unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class='glass-card' style='margin-bottom: 30px; border-left: 4px solid #818CF8;'>
                <h3 style='margin: 0 0 10px 0; color: #F8FAFC; font-weight: 800; font-size: 1.6rem;'>🧠 Neural Decision Transparency</h3>
                <p style='color: #94A3B8; margin: 0; font-size: 1.1rem; line-height: 1.6;'>
                    SHAP (SHapley Additive exPlanations) unpacks the machine learning model's black box. This waterfall chart visualizes exactly how much each specific input feature shifted your final predicted score relative to the baseline.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.spinner("Decoding Shapley Matrices..."):
            try:
                # Set matplotlib dark theme
                plt.style.use("dark_background")
                plt.rcParams.update(
                    {
                        "axes.facecolor": "none",
                        "figure.facecolor": "none",
                        "text.color": "#E2E8F0",
                        "axes.labelcolor": "#94A3B8",
                        "xtick.color": "#64748B",
                        "ytick.color": "#64748B",
                        "grid.color": "rgba(255,255,255,0.1)",
                    }
                )

                explainer = shap.Explainer(model)
                shap_values = explainer(st.session_state.transformed_input)

                fig, ax = plt.subplots(figsize=(10, 6))

                try:
                    feature_names = preprocessor.get_feature_names_out()
                    shap_values.feature_names = feature_names
                except Exception:
                    pass

                # Use matplotlib settings that look good on dark backgrounds
                shap.plots.waterfall(shap_values[0], show=False)

                # Make SHAP figure transparent to match background
                fig.patch.set_alpha(0.0)
                ax.patch.set_alpha(0.0)

                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.pyplot(fig, clear_figure=True, transparent=True)
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(
                    f"SHAP Matrix Generation Error: Ensure your model type is supported by SHAP Explainers. Details: {e}"
                )

        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================
    # TAB 4: AI STRATEGY ENGINE
    # ==========================
    with ai:
        st.markdown(
            "<div class='animate-in' style='margin-top: 25px;'>", unsafe_allow_html=True
        )

        if st.session_state.ai_advice is None:
            with st.spinner("Initializing Gemini Generative Core..."):
                st.session_state.ai_advice = advisor.generate_advice(
                    st.session_state.prediction, st.session_state.weak_features
                )

        st.markdown('<div class="ai-report-container">', unsafe_allow_html=True)
        st.markdown(st.session_state.ai_advice)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

else:
    # Stylized Standby Screen
    st.markdown(
        """
        <div class='animate-in' style='text-align: center; padding: 120px 40px; margin-top: 40px; display: flex; flex-direction: column; justify-content: center; align-items: center;'>
            <div style='font-size: 5rem; margin-bottom: 25px; animation: pulse 3s infinite; text-shadow: 0 0 30px rgba(56, 189, 248, 0.5);'>🌌</div>
            <h3 style='color: #F8FAFC; font-size: 2.2rem; font-family: "Outfit", sans-serif; font-weight: 800; margin: 0 0 15px 0; letter-spacing: 1px;'>AWAITING TELEMETRY</h3>
            <p style='color: #64748B; max-width: 600px; margin: 0 auto; font-size: 1.2rem; line-height: 1.6;'>
                System standing by. Please configure the demographic vectors and skill parameters in the command sidebar, then execute the prediction sequence.
            </p>
        </div>
        <style>
            @keyframes pulse {
                0% { transform: scale(1); filter: brightness(1); }
                50% { transform: scale(1.15); filter: brightness(1.2); }
                100% { transform: scale(1); filter: brightness(1); }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
