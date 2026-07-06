print("🔥🔥🔥 NEW RAG_ADVISOR.PY FILE LOADED 🔥🔥🔥")


import os
import streamlit as st
import google.generativeai as genai



class RAGAdvisor:


    def __init__(self):


        print("🤖 RAGAdvisor object created")


        self.api_key = None


        # ==========================
        # LOAD API KEY
        # ==========================

        try:

            self.api_key = st.secrets["GEMINI_API_KEY"]

            print("✅ GEMINI KEY LOADED FROM STREAMLIT SECRETS")


        except Exception:


            self.api_key = os.getenv(
                "GEMINI_API_KEY"
            )


            if self.api_key:

                print("✅ GEMINI KEY LOADED FROM ENVIRONMENT")


            else:

                print("❌ NO GEMINI API KEY FOUND")



        if self.api_key:


            genai.configure(
                api_key=self.api_key
            )


            # IMPORTANT:
            # use current supported model

            self.model = genai.GenerativeModel(
                "gemini-2.0-flash"
            )


        else:


            self.model = None




    def generate_advice(
        self,
        prediction,
        weak_features
    ):


        print("🚀 GEMINI FUNCTION CALLED 🚀")


        if self.model is None:


            return """
            ⚠️ Gemini API key missing.

            Add GEMINI_API_KEY in:

            Streamlit Cloud
            → Manage App
            → Settings
            → Secrets
            """



        try:


            print("🌐 Sending request to Gemini...")


            prompt = f"""

You are an expert AI academic advisor.

Analyze this student's performance.


Student predicted performance:

{prediction}


Weak areas:

{weak_features}



Give a detailed personalized improvement plan.


Include:


1. Performance Analysis

2. Weakness Explanation

3. Study Roadmap

4. Daily Routine

5. Resources

6. Motivation


Give a complete answer.

            """



            response = self.model.generate_content(
                prompt
            )


            print("✅ GEMINI RESPONSE RECEIVED")


            return response.text



        except Exception as e:


            print("❌ GEMINI FAILED")
            print(e)


            return f"""

⚠️ Gemini API Error


Actual Error:


{e}


Check:

1. API quota

2. Gemini model availability

3. Streamlit secrets

            """


