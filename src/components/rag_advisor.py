import os

from dotenv import load_dotenv

import google.generativeai as genai



# ==========================
# LOAD ENVIRONMENT VARIABLES
# ==========================

load_dotenv()



# ==========================
# RAG ADVISOR CLASS
# ==========================

class RAGAdvisor:


    def __init__(
        self
    ):


        api_key = os.getenv(
            "GOOGLE_API_KEY"
        )


        genai.configure(
            api_key=api_key
        )


        self.model = genai.GenerativeModel(
            "gemini-2.0-flash-lite"
        )



    # ==========================
    # GENERATE AI ADVICE
    # ==========================

    def generate_advice(
        self,
        prediction,
        weak_features
    ):


        prompt = f"""


You are an AI Student Performance Advisor.


Analyze the student's academic performance.


Student Prediction Score:
{round(prediction,2)}


Weak Areas:
{weak_features}



Provide a detailed improvement report:


1. Performance Analysis

Explain current performance level.



2. Weakness Explanation

Explain why these weak areas affect performance.



3. Personalized Improvement Plan

Give practical steps.



4. Study Roadmap

Give weekly improvement strategy.



5. Motivation

End with encouraging advice.


Keep response structured and detailed.


"""



        # ==========================
        # TRY GEMINI RESPONSE
        # ==========================

        try:


            response = self.model.generate_content(
                prompt
            )


            return response.text



        # ==========================
        # FALLBACK RESPONSE
        # ==========================

        except Exception as error:


            weak_text = (
                ", ".join(
                    weak_features
                )

                if weak_features

                else

                "No major weaknesses detected 🎉"
            )



            return f"""

## 📊 AI Student Performance Report


### 1. Performance Analysis


Your predicted performance score is:

### ⭐ {round(prediction,2)}%


This indicates your current academic performance level.

You are progressing well, but continuous improvement can increase your score further.



---


## 2. Weakness Explanation


Detected Weak Areas:


**{weak_text}**


These areas have the highest impact on improving future performance.



---


## 3. Personalized Improvement Plan


Recommended actions:


- Create a fixed study routine

- Practice difficult topics regularly

- Review mistakes after every test

- Improve problem solving speed

- Track weekly progress



---


## 4. Weekly Study Roadmap


### Week 1

- Identify weak concepts

- Revise fundamentals


### Week 2

- Practice questions daily

- Improve accuracy


### Week 3

- Attempt mock tests

- Analyze mistakes


### Week 4

- Revision

- Performance optimization



---


## 5. Final Recommendation 🚀


Your ML prediction pipeline is running successfully.


The Generative AI service limit is temporarily reached, so this backup advisor generated your improvement plan.


Continue consistent improvement and your performance score can increase.

"""