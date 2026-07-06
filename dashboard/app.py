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
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="AI Student Success Platform",
    page_icon="🎓",
    layout="wide"
)


st.title(
    "🎓 AI Student Success Platform"
)


st.write(
    "End-to-End ML + Explainable AI + Generative AI Student Advisor"
)



# ==========================
# LOAD ARTIFACTS
# ==========================

@st.cache_resource
def load_artifacts():


    with open(
        "artifacts/models/model.pkl",
        "rb"
    ) as file:

        model = dill.load(
            file
        )


    with open(
        "artifacts/preprocessors/preprocessor.pkl",
        "rb"
    ) as file:

        preprocessor = dill.load(
            file
        )


    return model, preprocessor



model, preprocessor = load_artifacts()



# ==========================
# SIDEBAR
# ==========================

st.sidebar.header(
    "🎯 Student Information"
)



gender = st.sidebar.selectbox(
    "Gender",
    [
        "male",
        "female"
    ]
)


race = st.sidebar.selectbox(
    "Race/Ethnicity",
    [
        "group A",
        "group B",
        "group C",
        "group D",
        "group E"
    ]
)


parental = st.sidebar.selectbox(
    "Parental Education",
    [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ]
)


lunch = st.sidebar.selectbox(
    "Lunch",
    [
        "standard",
        "free/reduced"
    ]
)


test_preparation_course = st.sidebar.selectbox(
    "Test Preparation",
    [
        "completed",
        "none"
    ]
)


math_score = st.sidebar.slider(
    "Math Score",
    0,
    100,
    70
)


reading_score = st.sidebar.slider(
    "Reading Score",
    0,
    100,
    70
)


writing_score = st.sidebar.slider(
    "Writing Score",
    0,
    100,
    70
)



# ==========================
# BUTTON
# ==========================

if st.button(
    "Predict Student Performance 🚀"
):


    input_data = pd.DataFrame(

        {

            "gender":
            [
                gender
            ],


            "race/ethnicity":
            [
                race
            ],


            "parental level of education":
            [
                parental
            ],


            "lunch":
            [
                lunch
            ],


            "test preparation course":
            [
                test_preparation_course
            ],


            "math score":
            [
                math_score
            ],


            "reading score":
            [
                reading_score
            ],


            "writing score":
            [
                writing_score
            ]

        }

    )



    transformed_data = preprocessor.transform(
        input_data
    )


    prediction = model.predict(
        transformed_data
    )[0]



    # ==========================
    # KPI DASHBOARD
    # ==========================

    st.subheader(
        "📊 Performance Dashboard"
    )


    avg_score = round(
        (
            math_score
            +
            reading_score
            +
            writing_score
        )
        /
        3,
        2
    )



    weak_features = []


    if math_score < 60:

        weak_features.append(
            "Math"
        )


    if reading_score < 60:

        weak_features.append(
            "Reading"
        )


    if writing_score < 60:

        weak_features.append(
            "Writing"
        )


    if test_preparation_course == "none":

        weak_features.append(
            "Test Preparation"
        )



    col1, col2, col3 = st.columns(
        3
    )


    col1.metric(
        "AI Prediction",
        round(
            prediction,
            2
        )
    )


    col2.metric(
        "Average Score",
        avg_score
    )


    col3.metric(
        "Weak Areas",
        len(
            weak_features
        )
    )



    # ==========================
    # BAR CHART
    # ==========================

    score_df = pd.DataFrame(

        {

            "Subject":
            [
                "Math",
                "Reading",
                "Writing"
            ],


            "Score":
            [
                math_score,
                reading_score,
                writing_score
            ]

        }

    )


    fig = px.bar(
        score_df,
        x="Subject",
        y="Score",
        text="Score",
        title="Subject Wise Score Analysis"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



    # ==========================
    # GAUGE CHART
    # ==========================

    gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=prediction,

            title={
                "text":
                "AI Success Prediction"
            },

            gauge={

                "axis":
                {

                    "range":
                    [
                        0,
                        100
                    ]

                }

            }

        )

    )


    st.plotly_chart(
        gauge,
        use_container_width=True
    )



    # ==========================
    # PIE CHART
    # ==========================

    pie_data = pd.DataFrame(

        {

            "Status":
            [
                "Achieved",
                "Remaining"
            ],


            "Value":
            [
                prediction,
                100 - prediction
            ]

        }

    )


    pie = px.pie(

        pie_data,

        names="Status",

        values="Value",

        title="Performance Completion"

    )


    st.plotly_chart(
        pie,
        use_container_width=True
    )



    # ==========================
    # WEAK AREAS
    # ==========================

    st.subheader(
        "📉 Weak Areas"
    )


    if weak_features:


        st.warning(
            weak_features
        )


    else:


        st.success(
            "No major weak areas detected 🎉"
        )



    # ==========================
    # GEMINI
    # ==========================

    st.subheader(
        "🤖 AI Improvement Advisor"
    )


    with st.spinner(
        "Generating personalized roadmap..."
    ):


        advisor = RAGAdvisor()


        advice = advisor.generate_advice(
            prediction,
            weak_features
        )


    st.write(
        advice
    )
