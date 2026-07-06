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

from src.components.rag_advisor import RAGAdvisor


# ==========================
# RESPONSIVE PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="AI Student Success Platform",
    page_icon="🎓",
    layout="wide"  # Uses the full available browser width fluidly
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


# ==========================
# HEADER
# ==========================
st.title("🎓 AI Student Success Platform")
st.write("End-to-End ML + Explainable AI + Generative AI Student Advisor")


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
# SIDEBAR INPUT (Collapsible on Mobile)
# ==========================
st.sidebar.header("🎯 Student Information")

gender = st.sidebar.selectbox("Gender", ["male", "female"])
race = st.sidebar.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parental = st.sidebar.selectbox("Parental Education", ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
lunch = st.sidebar.selectbox("Lunch", ["standard", "free/reduced"])
test_preparation_course = st.sidebar.selectbox("Test Preparation", ["completed", "none"])

math_score = st.sidebar.slider("Math Score", 0, 100, 70)
reading_score = st.sidebar.slider("Reading Score", 0, 100, 70)
writing_score = st.sidebar.slider("Writing Score", 0, 100, 70)


# ==========================
# PREDICTION TRIGGER
# ==========================
if st.sidebar.button("Predict Student Performance 🚀", use_container_width=True):
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

    transformed_data = preprocessor.transform(input_data)
    pred = model.predict(transformed_data)[0]
    st.session_state.prediction = round(pred, 2)

    # Weakness Logic
    weak = []
    if math_score < 60:
        weak.append("Math")
    if reading_score < 60:
        weak.append("Reading")
    if writing_score < 60:
        weak.append("Writing")
    if test_preparation_course == "none":
        weak.append("Test Preparation")
        
    st.session_state.weak_features = weak
    st.session_state.avg_score = round((math_score + reading_score + writing_score) / 3, 2)
    
    # Reset AI text buffer to force a fresh request when parameters modify
    st.session_state.ai_advice = None


# ==========================
# DISPLAY INTERFACE
# ==========================
if st.session_state.prediction_triggered:

    # Tabs scale horizontally on desktop and scroll gracefully on touch devices
    dashboard, charts, ai = st.tabs([
        "📊 Dashboard", 
        "📈 Analytics", 
        "🤖 AI Advisor"
    ])

    # ==========================
    # DASHBOARD
    # ==========================
    with dashboard:
        # Columns break down into individual stacked blocks cleanly on narrow phones
        col1, col2, col3 = st.columns([1, 1, 1])
        col1.metric("AI Prediction", f"{st.session_state.prediction}%")
        col2.metric("Average Score", f"{st.session_state.avg_score}%")
        col3.metric("Weak Areas Count", len(st.session_state.weak_features))

        st.divider()

        if st.session_state.weak_features:
            st.warning(f"**Areas of Improvement Recommended:** {', '.join(st.session_state.weak_features)}")
        else:
            st.success("Excellent! No major academic vulnerabilities detected 🎉")

    # ==========================
    # ANALYTICS (Device-Width Aware)
    # ==========================
    with charts:
        score_df = pd.DataFrame({
            "Subject": ["Math", "Reading", "Writing"],
            "Score": [math_score, reading_score, writing_score]
        })

        # Bar Graph
        bar = px.bar(
            score_df,
            x="Subject",
            y="Score",
            text="Score",
            title="Subject Performance"
        )
        bar.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(bar, use_container_width=True)

        # Gauge Chart
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=st.session_state.prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={"text": "Success Probability"}
            )
        )
        gauge.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
        st.plotly_chart(gauge, use_container_width=True)

        # Radar Grid Layout
        radar = go.Figure()
        radar.add_trace(
            go.Scatterpolar(
                r=[math_score, reading_score, writing_score],
                theta=["Math", "Reading", "Writing"],
                fill="toself"
            )
        )
        radar.update_layout(
            title="Skill Radar Analysis",
            margin=dict(l=40, r=40, t=40, b=40),
            height=350
        )
        st.plotly_chart(radar, use_container_width=True)

    # ==========================
    # AI ADVISOR
    # ==========================
    with ai:
        st.subheader("🤖 Personalized AI Improvement Advisor")

        if st.session_state.ai_advice is None:
            with st.spinner("Generating complete AI roadmap..."):
                st.session_state.ai_advice = advisor.generate_advice(
                    st.session_state.prediction,
                    st.session_state.weak_features
                )

        # Container elements guarantee fluid textual line-wrapping on compact smartphone viewports
        with st.container():
            st.markdown(st.session_state.ai_advice)

else:
    st.info("👈 Open the sidebar menu, configure student metrics, and click 'Predict Student Performance' to run evaluations.")
