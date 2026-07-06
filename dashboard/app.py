import sys
import os

# ==========================
# FIX PROJECT IMPORT PATH
# ==========================
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import streamlit as st
import pandas as pd
import dill
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

from src.components.rag_advisor import RAGAdvisor

# ==========================
# RESPONSIVE PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="AI Student Success Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# PREMIUM GLOBAL CSS INJECTION
# ==========================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }
        
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animated-container {
            animation: fadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        .metric-card {
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            position: relative;
            overflow: hidden;
        }
        .metric-card:hover {
            transform: translateY(-5px) scale(1.02);
            box-shadow: 0 15px 30px rgba(0,0,0,0.15) !important;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            padding-bottom: 5px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            color: #64748B;
            border: 1px solid transparent;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #F1F5F9;
            color: #1E293B;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1E293B !important;
            color: white !important;
            box-shadow: 0 4px 12px rgba(30,41,59,0.2);
        }
        
        .stSlider > div > div > div { background-color: #3B82F6 !important; }
        div[data-baseweb="select"] > div { border-radius: 8px !important; }

        .ai-report-container {
            background-color: white;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.04);
            border: 1px solid #F1F5F9;
            line-height: 1.7;
            font-size: 1.05rem;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)


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
    <div class='animated-container' style='background: linear-gradient(135deg, #020617, #1E293B); padding: 40px; border-radius: 20px; margin-bottom: 35px; box-shadow: 0 20px 40px -10px rgba(0,0,0,0.4); position: relative; overflow: hidden;'>
        <div style='position: absolute; top: -50px; right: -50px; width: 200px; height: 200px; background: rgba(56, 189, 248, 0.2); filter: blur(50px); border-radius: 50%;'></div>
        <h1 style='color: #F8FAFC; margin: 0; font-size: 2.8rem; font-weight: 800; letter-spacing: -0.5px;'>🎓 AI Student Success Platform</h1>
        <p style='color: #94A3B8; margin: 12px 0 0 0; font-size: 1.2rem; font-weight: 400;'>Next-generation predictive analytics & personalized academic telemetry.</p>
    </div>
    """, 
    unsafe_allow_html=True
)


# ==========================
# LOAD ARTIFACTS & RESOURCES
# ==========================
@st.cache_resource
def load_artifacts():
    with open("artifacts/models/model.pkl", "rb") as file:
        model = dill.load(file)

    with open("artifacts/preprocessors/preprocessor.pkl", "rb") as file:
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
    st.markdown("<h3 style='color: #0F172A; font-weight: 700; margin-bottom: 20px;'>⚙️ Configuration Parameters</h3>", unsafe_allow_html=True)
    
    gender = st.selectbox("Gender Identity", ["male", "female"])
    race = st.selectbox("Demographic Group", ["group A", "group B", "group C", "group D", "group E"])
    parental = st.selectbox("Parental Education", ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
    lunch = st.selectbox("Meal Plan", ["standard", "free/reduced"])
    test_preparation_course = st.selectbox("Prep Course Completion", ["completed", "none"])

    st.markdown("<hr style='border: none; height: 1px; background: linear-gradient(90deg, transparent, #E2E8F0, transparent); margin: 30px 0;'/>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='color: #0F172A; font-weight: 700;'>📊 Current Telemetrics</h3>", unsafe_allow_html=True)
    math_score = st.slider("📐 Quantitative Logic (Math)", 0, 100, 70)
    reading_score = st.slider("📖 Comprehension (Reading)", 0, 100, 70)
    writing_score = st.slider("✍️ Expression (Writing)", 0, 100, 70)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    predict_clicked = st.button("Initialize Prediction Sequence 🚀", use_container_width=True, type="primary")


# ==========================
# PREDICTION ENGINE
# ==========================
if predict_clicked:
    st.session_state.prediction_triggered = True
    
    input_data = pd.DataFrame({
        "gender": [gender],
        "race/ethnicity": [race],
        "parental level of education": [parental],
        "lunch": [lunch],
        "test preparation course": [test_preparation_course],
        "math score": [math_score],
        "reading score": [reading_score],
        "writing score": [writing_score]
    })

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

    dashboard, charts, explain, ai = st.tabs([
        "⊞ Executive Dashboard", 
        "🌌 Advanced Visualizations", 
        "🧠 AI Explainability",
        "🤖 AI Strategy Engine"
    ])

    # ==========================
    # TAB 1: EXECUTIVE DASHBOARD
    # ==========================
    with dashboard:
        st.markdown("<div class='animated-container' style='margin-top: 25px;'>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown(
                f"""
                <div class='metric-card' style='background: linear-gradient(135deg, #0EA5E9, #2563EB); padding: 30px; border-radius: 20px; color: white; box-shadow: 0 10px 25px rgba(37,99,235,0.2); display: flex; flex-direction: column; justify-content: center; align-items: center;'>
                    <div style='background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 30px; font-size: 0.85rem; font-weight: 700; letter-spacing: 1px; margin-bottom: 15px;'>MODEL FORECAST</div>
                    <h2 style='margin: 0; font-size: 3.5rem; font-weight: 800; line-height: 1;'>{st.session_state.prediction:.2f}<span style='font-size: 1.5rem; opacity: 0.8;'>%</span></h2>
                </div>
                """, unsafe_allow_html=True
            )
            
        with col2:
            st.markdown(
                f"""
                <div class='metric-card' style='background: linear-gradient(135deg, #10B981, #059669); padding: 30px; border-radius: 20px; color: white; box-shadow: 0 10px 25px rgba(16,185,129,0.2); display: flex; flex-direction: column; justify-content: center; align-items: center;'>
                    <div style='background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 30px; font-size: 0.85rem; font-weight: 700; letter-spacing: 1px; margin-bottom: 15px;'>CURRENT AVERAGE</div>
                    <h2 style='margin: 0; font-size: 3.5rem; font-weight: 800; line-height: 1;'>{st.session_state.avg_score:.2f}<span style='font-size: 1.5rem; opacity: 0.8;'>%</span></h2>
                </div>
                """, unsafe_allow_html=True
            )
            
        with col3:
            risk_gradient = "linear-gradient(135deg, #F43F5E, #E11D48)" if len(st.session_state.weak_features) > 0 else "linear-gradient(135deg, #8B5CF6, #6D28D9)"
            risk_shadow = "rgba(244,63,94,0.2)" if len(st.session_state.weak_features) > 0 else "rgba(139,92,246,0.2)"
            st.markdown(
                f"""
                <div class='metric-card' style='background: {risk_gradient}; padding: 30px; border-radius: 20px; color: white; box-shadow: 0 10px 25px {risk_shadow}; display: flex; flex-direction: column; justify-content: center; align-items: center;'>
                    <div style='background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 30px; font-size: 0.85rem; font-weight: 700; letter-spacing: 1px; margin-bottom: 15px;'>SYSTEM RISKS</div>
                    <h2 style='margin: 0; font-size: 3.5rem; font-weight: 800; line-height: 1;'>{len(st.session_state.weak_features)}</h2>
                </div>
                """, unsafe_allow_html=True
            )

        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)

        if st.session_state.weak_features:
            weak_items_html = "".join([f"<div style='background: white; padding: 12px 20px; margin: 10px 0; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); font-weight: 600; color: #BE123C; display: flex; align-items: center; gap: 10px;'><span style='background: #FFE4E6; width: 24px; height: 24px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 12px;'>⚠️</span> {item}</div>" for item in st.session_state.weak_features])
            st.markdown(
                f"""
                <div class='metric-card' style='background: #FFF1F2; padding: 30px; border-radius: 20px; border: 1px solid #FECDD3;'>
                    <h3 style='margin: 0 0 15px 0; color: #9F1239; font-weight: 800; font-size: 1.4rem;'>Critical Diagnostics Flagged</h3>
                    {weak_items_html}
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class='metric-card' style='background: #F0FDF4; padding: 35px; border-radius: 20px; border: 1px solid #BBF7D0; text-align: center;'>
                    <div style='font-size: 3rem; margin-bottom: 15px;'>✨</div>
                    <h3 style='margin: 0; color: #166534; font-weight: 800; font-size: 1.5rem;'>Optimal Performance Trajectory</h3>
                    <p style='color: #15803D; margin: 10px 0 0 0; font-size: 1.1rem;'>All baseline metrics are stabilized. No structural vulnerabilities detected.</p>
                </div>
                """, unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================
    # TAB 2: ADVANCED VISUALIZATIONS
    # ==========================
    with charts:
        st.markdown("<div class='animated-container' style='margin-top: 25px;'>", unsafe_allow_html=True)
        
        score_df = pd.DataFrame({
            "Subject": ["Math", "Reading", "Writing"],
            "Score": [math_score, reading_score, writing_score]
        })

        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            st.markdown("<div class='metric-card' style='background: white; padding: 20px; border-radius: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.03); border: 1px solid #F1F5F9;'>", unsafe_allow_html=True)
            bar = px.bar(
                score_df,
                x="Subject",
                y="Score",
                text="Score",
                title="Capability Matrix Distribution",
                color="Subject",
                color_discrete_map={"Math": "#3B82F6", "Reading": "#10B981", "Writing": "#8B5CF6"}
            )
            bar.update_traces(
                textposition='inside', 
                textfont=dict(size=16, color="white", family="Inter", weight="bold"),
                marker_line_width=0
            )
            bar.update_layout(
                showlegend=False, 
                margin=dict(l=10, r=10, t=50, b=10), 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                title_font=dict(size=18, color="#1E293B", family="Inter")
            )
            st.plotly_chart(bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with chart_col2:
            st.markdown("<div class='metric-card' style='background: white; padding: 20px; border-radius: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.03); border: 1px solid #F1F5F9;'>", unsafe_allow_html=True)
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=st.session_state.prediction,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={"text": "System Health Velocity", "font": {"size": 18, "color": "#1E293B", "family": "Inter"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 0},
                        'bar': {'color': "#0EA5E9", 'thickness': 0.85},
                        'bgcolor': "#F1F5F9",
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 55], 'color': '#FFE4E6'},
                            {'range': [55, 75], 'color': '#FEF08A'},
                            {'range': [75, 100], 'color': '#D1FAE5'}
                        ]
                    }
                )
            )
            gauge.update_layout(margin=dict(l=20, r=20, t=60, b=10), height=320, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(gauge, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
        
        radar_center_col, _ = st.columns([1.5, 1])
        with radar_center_col:
            st.markdown("<div class='metric-card' style='background: white; padding: 30px; border-radius: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.03); border: 1px solid #F1F5F9;'>", unsafe_allow_html=True)
            radar = go.Figure()
            radar.add_trace(
                go.Scatterpolar(
                    r=[math_score, reading_score, writing_score, math_score],
                    theta=["Quantitative", "Comprehension", "Synthesizing", "Quantitative"],
                    fill="toself",
                    fillcolor="rgba(14, 165, 233, 0.15)",
                    line=dict(color="#0EA5E9", width=3, shape='spline')
                )
            )
            radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="#F1F5F9", linecolor="rgba(0,0,0,0)"),
                    angularaxis=dict(gridcolor="#F1F5F9", linecolor="rgba(0,0,0,0)")
                ),
                title="Symmetry & Alignment Profile",
                margin=dict(l=50, r=50, t=70, b=50),
                height=450,
                paper_bgcolor='rgba(0,0,0,0)',
                title_font=dict(size=20, color="#1E293B", family="Inter")
            )
            st.plotly_chart(radar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================
    # TAB 3: AI EXPLAINABILITY (SHAP)
    # ==========================
    with explain:
        st.markdown("<div class='animated-container' style='margin-top: 25px;'>", unsafe_allow_html=True)
        
        st.markdown(
            """
            <div style='background: white; padding: 30px; border-radius: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.03); border: 1px solid #F1F5F9; margin-bottom: 25px;'>
                <h3 style='margin: 0 0 10px 0; color: #1E293B; font-weight: 800;'>🧠 Neural Decision Transparency</h3>
                <p style='color: #64748B; margin: 0; font-size: 1.05rem;'>SHAP (SHapley Additive exPlanations) unpacks the machine learning model's black box. This waterfall chart visualizes exactly how much each specific input feature increased (red) or decreased (blue) your final predicted score from the model's base expected value.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

        with st.spinner("Executing Shapley Additive Explanations on live neural vectors..."):
            try:
                # Initialize SHAP explainer
                # For XGBoost/Tree models, TreeExplainer is fast and highly optimized
                explainer = shap.Explainer(model)
                
                # Calculate SHAP values for the specific transformed input
                shap_values = explainer(st.session_state.transformed_input)
                
                # Setup Matplotlib figure for the SHAP Waterfall Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Attempt to get feature names if the preprocessor exposes them, 
                # otherwise SHAP defaults to feature numbers which still demonstrates the mathematical logic.
                try:
                    feature_names = preprocessor.get_feature_names_out()
                    shap_values.feature_names = feature_names
                except Exception:
                    pass # Fallback to numerical indexing if pipeline doesn't support get_feature_names_out()

                # Generate the Waterfall plot
                shap.plots.waterfall(shap_values[0], show=False)
                
                # Inject plot into Streamlit UI
                st.pyplot(fig, clear_figure=True)

            except Exception as e:
                st.error(f"SHAP Matrix Generation Error: Ensure your model type is supported by SHAP Explainers. Details: {e}")

        st.markdown("</div>", unsafe_allow_html=True)


    # ==========================
    # TAB 4: AI STRATEGY ENGINE
    # ==========================
    with ai:
        st.markdown("<div class='animated-container' style='margin-top: 25px;'>", unsafe_allow_html=True)
        
        if st.session_state.ai_advice is None:
            with st.spinner("Initializing generative strategy engine..."):
                st.session_state.ai_advice = advisor.generate_advice(
                    st.session_state.prediction,
                    st.session_state.weak_features
                )

        st.markdown('<div class="ai-report-container">', unsafe_allow_html=True)
        st.markdown(st.session_state.ai_advice)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # Stylized Standby Screen
    st.markdown(
        """
        <div class='animated-container' style='text-align: center; padding: 100px 40px; background: rgba(248, 250, 252, 0.8); backdrop-filter: blur(10px); border: 2px dashed #CBD5E1; border-radius: 24px; margin-top: 30px;'>
            <div style='font-size: 4rem; margin-bottom: 20px; animation: pulse 2s infinite;'>⚡</div>
            <h3 style='color: #0F172A; font-size: 1.8rem; font-weight: 800; margin: 0;'>Systems on Standby</h3>
            <p style='color: #64748B; max-width: 550px; margin: 15px auto 0 auto; font-size: 1.15rem; line-height: 1.6;'>Awaiting data inputs. Please configure the demographic and grade parameters in the sidebar, then execute the prediction sequence.</p>
        </div>
        <style>
            @keyframes pulse {
                0% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.1); opacity: 0.7; }
                100% { transform: scale(1); opacity: 1; }
            }
        </style>
        """, 
        unsafe_allow_html=True
    )