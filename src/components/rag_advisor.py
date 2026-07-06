# ==========================================
# AI Student Success Platform
# Gemini AI Improvement Advisor
# Streamlit Cloud + Local Compatible
# ==========================================

print("🔥🔥🔥 NEW RAG ADVISOR FILE LOADED 🔥🔥🔥")
import os
import streamlit as st
import google.generativeai as genai



class RAGAdvisor:


    def __init__(self):


        # ==============================
        # LOAD GEMINI API KEY
        # ==============================

        self.api_key = None


        # Streamlit Cloud Secrets

        try:


            if "GOOGLE_API_KEY" in st.secrets:


                self.api_key = st.secrets[
                    "GOOGLE_API_KEY"
                ]


            elif "GEMINI_API_KEY" in st.secrets:


                self.api_key = st.secrets[
                    "GEMINI_API_KEY"
                ]


        except Exception:


            pass



        # Local Environment

        if self.api_key is None:


            self.api_key = os.getenv(
                "GOOGLE_API_KEY"
            )


        if self.api_key is None:


            self.api_key = os.getenv(
                "GEMINI_API_KEY"
            )




        # ==============================
        # DEBUG LOGS
        # ==============================


        print(
            "================================="
        )


        print(
            "Gemini API Key Loaded:",
            bool(self.api_key)
        )


        print(
            "Using Model: gemini-2.0-flash-lite"
        )


        print(
            "================================="
        )




        if not self.api_key:


            raise Exception(
                "GOOGLE_API_KEY / GEMINI_API_KEY missing"
            )




        # ==============================
        # GEMINI CONFIG
        # ==============================


        genai.configure(
            api_key=self.api_key
        )



        self.model = genai.GenerativeModel(
            "gemini-2.0-flash-lite"
        )






    # ==================================
    # GENERATE AI RESPONSE
    # ==================================


    def generate_advice(
        self,
        prediction,
        weak_features
    ):


        print(
            "Calling Gemini API..."
        )



        prompt = f"""


You are an expert AI Student Success Advisor.


Analyze this student's academic performance.


Student Data:


Predicted Score:

{round(prediction,2)}%


Weak Areas:

{weak_features}




Generate a complete personalized report.



Include these sections:



1. 📊 Performance Analysis

- Explain current performance
- Strengths
- Academic condition



2. 📉 Weakness Explanation

- Explain every weak area
- Reason behind weakness
- How it impacts results



3. 🚀 Improvement Roadmap

Give:

- Daily improvement tasks
- Learning techniques
- Practice strategy



4. 📅 7 Day Study Plan

Create a practical weekly schedule.



5. 📚 Resources

Suggest:

- Study methods
- Online resources
- Practice ideas



6. 💡 Final Motivation

Give encouraging student advice.



Rules:

- Do NOT give generic answers
- Personalize based on scores
- Keep explanation detailed
- Format nicely


"""



        try:



            response = self.model.generate_content(
                prompt
            )



            print(
                "Gemini Response Received Successfully"
            )



            if response.text:


                return response.text


            else:


                return (
                    "Gemini returned an empty response."
                )





        except Exception as e:



            print(
                "================================="
            )


            print(
                "Gemini API ERROR:"
            )


            print(
                str(e)
            )


            print(
                "================================="
            )




            return f"""

⚠️ Gemini API Error


{str(e)}


Possible reasons:

1. Gemini free quota finished

2. API limit reached

3. Model temporarily unavailable


"""


