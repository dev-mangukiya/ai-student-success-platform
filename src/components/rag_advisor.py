# =====================================
# RAG + GEMINI AI STUDENT ADVISOR
# Streamlit Cloud Compatible
# =====================================


import os
import streamlit as st
import google.generativeai as genai



class RAGAdvisor:


    def __init__(self):


        # ==============================
        # LOAD GEMINI API KEY
        # ==============================

        try:

            self.api_key = st.secrets[
                "GOOGLE_API_KEY"
            ]


        except Exception:


            self.api_key = os.getenv(
                "GOOGLE_API_KEY"
            )



        # DEBUG FOR STREAMLIT LOGS

        print(
            "================================"
        )

        print(
            "GEMINI KEY FOUND:",
            self.api_key is not None
        )

        print(
            "================================"
        )



        if self.api_key is None:


            raise Exception(
                "GOOGLE_API_KEY NOT FOUND"
            )



        # ==============================
        # CONFIGURE GEMINI
        # ==============================


        genai.configure(
            api_key=self.api_key
        )


        self.model = genai.GenerativeModel(
            "gemini-2.0-flash-lite"
        )





    # ==============================
    # GENERATE AI ADVICE
    # ==============================


    def generate_advice(
        self,
        prediction,
        weak_features
    ):


        try:


            prompt = f"""

You are an expert AI Academic Performance Advisor.


Student Details:

Predicted Academic Performance:
{round(prediction,2)}%


Detected Weak Areas:
{weak_features}



Create a detailed personalized improvement report.


Include:


1. Performance Analysis
- Explain current level
- Strengths
- Academic risk


2. Weakness Explanation
- Explain each weak area
- Why it affects performance


3. Personalized Improvement Plan
- Daily actions
- Learning strategy


4. 7 Day Study Roadmap
- Day wise schedule


5. Motivation Advice
- Short encouraging advice


Make response detailed, structured and student friendly.

"""



            response = self.model.generate_content(
                prompt
            )



            print(
                "GEMINI RESPONSE GENERATED SUCCESSFULLY"
            )



            return response.text





        except Exception as e:



            print(
                "================================"
            )

            print(
                "GEMINI ERROR:"
            )

            print(
                e
            )

            print(
                "================================"
            )



            return f"""

⚠️ Gemini Failed

Actual Error:

{e}


Check:

1. Streamlit Secrets
2. API quota
3. Gemini API key validity

"""