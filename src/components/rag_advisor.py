import os
import streamlit as st
import google.generativeai as genai

print("🔥🔥🔥 NEW RAG_ADVISOR.PY FILE LOADED 🔥🔥🔥", flush=True)

class RAGAdvisor:

    def __init__(self):
        print("🤖 RAGAdvisor object created", flush=True)
        self.api_key = None
        self.model = None

        # ==========================
        # LOAD GEMINI API KEY
        # ==========================
        try:
            self.api_key = st.secrets["GEMINI_API_KEY"]
            print("✅ GEMINI KEY LOADED FROM STREAMLIT SECRETS", flush=True)
        except Exception as e:
            print("⚠️ Streamlit secrets not found, checking ENV", flush=True)
            self.api_key = os.getenv("GEMINI_API_KEY")
            
            if self.api_key:
                print("✅ GEMINI KEY LOADED FROM ENVIRONMENT", flush=True)
            else:
                print("❌ GEMINI API KEY NOT FOUND", flush=True)

        # ==========================
        # CONFIGURE GEMINI
        # ==========================
        if self.api_key:
            genai.configure(api_key=self.api_key)
            # Utilizing flash-lite to manage free-tier quotas effectively
            self.model = genai.GenerativeModel("gemini-2.0-flash-lite")
            print("✅ GEMINI MODEL INITIALIZED: gemini-2.0-flash-lite", flush=True)


    def generate_advice(self, prediction, weak_features):
        print("🚀 GEMINI FUNCTION CALLED 🚀", flush=True)

        if self.model is None:
            return """
⚠️ Gemini API key missing.

Add key:
Streamlit Cloud -> Manage App -> Settings -> Secrets

Format:
GEMINI_API_KEY="your_key_here"
"""

        try:
            print("🌐 Sending request to Gemini...", flush=True)
            
            prompt = f"""
You are an expert AI academic mentor.

Analyze this student:
Predicted Academic Performance: {round(prediction, 2)}%
Weak Areas: {weak_features}

Create a COMPLETE improvement report.
Include:
1. 📊 Performance Analysis - Explain current level.
2. 📉 Weakness Explanation - Explain why these areas affect results.
3. 📚 Personalized Study Roadmap - Give weekly improvement plan.
4. 🕒 Daily Routine - Morning, afternoon, evening plan.
5. 🎯 Score Improvement Strategy - How to increase marks.
6. 🚀 Motivation Advice - Encourage the student.

Give a detailed answer. Do not keep it short.
"""

            response = self.model.generate_content(prompt)
            print("✅ GEMINI RESPONSE RECEIVED SUCCESSFULLY", flush=True)

            if response.text:
                return response.text
            else:
                return "Gemini returned an empty response."

        except Exception as e:
            print("❌ GEMINI CALL FAILED", flush=True)
            print(f"Error Details: {e}", flush=True)
            
            return f"""
⚠️ Gemini API Error

The app encountered an error trying to process your request. 

**Actual Error Output:**
`{e}`

**Troubleshooting Steps:**
1. Check if your API Quota has reset (Free tier resets requests per minute).
2. Verify that your API Key is valid inside Streamlit Secrets.
3. If you keep hitting limits, consider switching to a pay-as-you-go tier in Google AI Studio.
"""


