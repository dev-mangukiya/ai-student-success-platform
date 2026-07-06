# ==================================
# Gemini AI Student Advisor
# ==================================

import os
import streamlit as st
import google.generativeai as genai


class RAGAdvisor:


    def __init__(self):


        self.api_key = None


        # Streamlit Cloud Secrets
        if "GOOGLE_API_KEY" in st.secrets:

            self.api_key = st.secrets["GOOGLE_API_KEY"]


        elif "GEMINI_API_KEY" in st.secrets:

            self.api_key = st.secrets["GEMINI_API_KEY"]



        # Local .env fallback
        else:

            self.api_key = os.getenv(
                "GOOGLE_API_KEY"
            )


        print(
            "================================"
        )

        print(
            "Gemini Key Loaded:",
            bool(self.api_key)
        )

        print(
            "Using Gemini Model"
        )

        print(
            "================================"
        )



        if not self.api_key:

            raise Exception(
                "No Gemini API key found"
            )


        genai.configure(
            api_key=self.api_key
        )


        self.model = genai.GenerativeModel(
            "gemini-1.5-flash"
        )





    def generate_advice(
        self,
        prediction,
        weak_features
    ):


        print(
            "Calling Gemini API..."
        )


        prompt = f"""

You are an AI Student Success Advisor.

Analyze this student:

Predicted Score:
{prediction}

Weak Areas:
{weak_features}


Generate:

1. Performance Analysis

2. Weakness Explanation

3. Personalized Improvement Plan

4. Weekly Study Roadmap

5. Recommended Resources


Make it detailed and personalized.

"""


        try:


            response = self.model.generate_content(
                prompt
            )


            print(
                "Gemini Response Received"
            )


            return response.text



        except Exception as e:


            print(
                "Gemini Failed:"
            )

            print(
                str(e)
            )


            return f"""

⚠️ Gemini API Error

{e}

"""