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
# LOAD MODEL + PREPROCESSOR
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
# SIDEBAR INPUTS
# ==========================

st.sidebar.header(
    "Enter Student Details"
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
    "Test Preparation Course",
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
# PREDICTION
# ==========================

if st.button(
    "Predict Student Performance 🚀"
):


    input_data = pd.DataFrame(
        {

            "gender": [
                gender
            ],

            "race/ethnicity": [
                race
            ],

            "parental level of education": [
                parental
            ],

            "lunch": [
                lunch
            ],

            "test preparation course": [
                test_preparation_course
            ],

            "math score": [
                math_score
            ],

            "reading score": [
                reading_score
            ],

            "writing score": [
                writing_score
            ]

        }
    )



    # ==========================
    # PREPROCESS DATA
    # ==========================

    transformed_data = preprocessor.transform(
        input_data
    )



    # ==========================
    # MODEL PREDICTION
    # ==========================

    prediction = model.predict(
        transformed_data
    )[0]



    st.success(
        f"Predicted Score: {round(prediction, 2)}"
    )



    # ==========================
    # WEAK FEATURE DETECTION
    # ==========================

    weak_features = []


    if math_score < 60:

        weak_features.append(
            "math score"
        )


    if reading_score < 60:

        weak_features.append(
            "reading score"
        )


    if writing_score < 60:

        weak_features.append(
            "writing score"
        )


    if test_preparation_course == "none":

        weak_features.append(
            "test preparation course"
        )



    st.subheader(
        "📉 Weak Areas"
    )


    if weak_features:

        st.write(
            weak_features
        )


    else:

        st.write(
            "No major weak areas detected 🎉"
        )



    # ==========================
    # GEMINI AI ADVISOR
    # ==========================

    with st.spinner(
        "Generating AI Improvement Plan..."
    ):


        advisor = RAGAdvisor()


        advice = advisor.generate_advice(
            prediction,
            weak_features
        )



    st.subheader(
        "🤖 AI Improvement Advisor"
    )


    st.write(
        advice
    )
