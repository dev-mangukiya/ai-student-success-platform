# ==========================================
# Gemini AI Student Improvement Advisor
# Streamlit Cloud + Local Compatible
# ==========================================


import os
import streamlit as st
import google.generativeai as genai



class RAGAdvisor:


    def __init__(self):


        # ==============================
        # GET GEMINI API KEY
        # ==============================

        self.api_key = None



        # Streamlit Cloud secrets

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





        # Local .env fallback

        if self.api_key is None:


            self.api_key = os.getenv(
                "GOOGLE_API_KEY"
            )



        # ==============================
        # DEBUG LOGS
        # ==============================


        print(
            "================================"
        )


        print(
            "Gemini Key Loaded:",
            bool(self.api_key)
        )


        print(
            "Gemini Model: gemini-2.0-flash"
        )


        print(
            "================================"
        )




        if not self.api_key:


            raise Exception(
                "Gemini API Key Missing"
            )





        # ==============================
        # CONFIGURE GEMINI
        # ==============================


        genai.configure(
            api_key=self.api_key
        )



        self.model = genai.GenerativeModel(
            "gemini-2.0-flash"
        )







    # ==================================
    # GENERATE STUDENT ADVICE
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


Student Information:


Predicted Performance Score:

{round(prediction,2)}%



Weak Areas:

{weak_features}





Generate a complete personalized report.



Include:



1. Performance Analysis

- Explain current performance level
- Mention strengths
- Mention risks



2. Weakness Explanation

- Explain each weak subject
- Why improvement is needed



3. Personalized Improvement Roadmap

- Daily habits
- Learning strategy
- Practice methods



4. 7-Day Study Plan

Create a day-wise schedule.



5. Recommended Resources

Suggest learning methods/tools.



6. Motivation

Give short motivational advice.



Make the answer detailed, practical, and student friendly.


"""



        try:



            response = self.model.generate_content(
                prompt
            )




            print(
                "Gemini Response Received Successfully"
            )




            return response.text





        except Exception as e:




            print(
                "================================"
            )


            print(
                "Gemini Failed:"
            )


            print(
                str(e)
            )


            print(
                "================================"
            )





            return f"""

⚠️ Gemini API Error


{str(e)}


"""

