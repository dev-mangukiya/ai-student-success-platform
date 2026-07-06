print("🔥🔥🔥 NEW RAG_ADVISOR.PY FILE LOADED 🔥🔥🔥")


import os
import streamlit as st
import google.generativeai as genai



class RAGAdvisor:


    def __init__(self):


        print("🤖 RAGAdvisor object created")


        self.api_key = None
        self.model = None


        # ==========================
        # LOAD GEMINI API KEY
        # ==========================

        try:


            self.api_key = st.secrets[
                "GEMINI_API_KEY"
            ]


            print(
                "✅ GEMINI KEY LOADED FROM STREAMLIT SECRETS"
            )


        except Exception as e:


            print(
                "⚠️ Streamlit secrets not found, checking ENV"
            )


            self.api_key = os.getenv(
                "GEMINI_API_KEY"
            )


            if self.api_key:


                print(
                    "✅ GEMINI KEY LOADED FROM ENVIRONMENT"
                )


            else:


                print(
                    "❌ GEMINI API KEY NOT FOUND"
                )




        # ==========================
        # CONFIGURE GEMINI
        # ==========================

        if self.api_key:


            genai.configure(
                api_key=self.api_key
            )


            self.model = genai.GenerativeModel(
                "gemini-2.0-flash-lite"
            )


            print(
                "✅ GEMINI MODEL INITIALIZED: gemini-2.0-flash-lite"
            )





    def generate_advice(
        self,
        prediction,
        weak_features
    ):


        print(
            "🚀 GEMINI FUNCTION CALLED 🚀"
        )


        if self.model is None:


            return """

⚠️ Gemini API key missing.

Add key:

Streamlit Cloud
→ Manage App
→ Settings
→ Secrets


Format:


GEMINI_API_KEY="your_key_here"

"""



        try:


            print(
                "🌐 Sending request to Gemini..."
            )


            prompt = f"""


You are an expert AI academic mentor.


Analyze this student:


Predicted Academic Performance:
{round(prediction,2)}%


Weak Areas:
{weak_features}



Create a COMPLETE improvement report.


Include:


1. 📊 Performance Analysis

Explain current level.


2. 📉 Weakness Explanation

Explain why these areas affect results.


3. 📚 Personalized Study Roadmap

Give weekly improvement plan.


4. 🕒 Daily Routine

Morning, afternoon, evening plan.


5. 🎯 Score Improvement Strategy

How to increase marks.


6. 🚀 Motivation Advice

Encourage the student.



Give detailed answer.
Do not keep it short.

"""


            response = self.model.generate_content(
                prompt
            )


            print(
                "✅ GEMINI RESPONSE RECEIVED"
            )


            if response.text:


                return response.text


            else:


                return "Gemini returned empty response."





        except Exception as e:


            print(
                "❌ GEMINI FAILED"
            )


            print(
                e
            )


            return f"""

⚠️ Gemini API Error


Actual Error:


{e}


Check:

1. API quota

2. API key

3. Model availability


"""


