import os

import google.generativeai as genai

from dotenv import load_dotenv



load_dotenv()



class RAGAdvisor:


    def __init__(
        self
    ):


        api_key = os.getenv(
            "GOOGLE_API_KEY"
        )


        if api_key is None:

            raise Exception(
                "GOOGLE_API_KEY not found"
            )


        genai.configure(
            api_key=api_key
        )


        self.model = genai.GenerativeModel(

            model_name=
            "gemini-2.0-flash-lite",

            generation_config={

                "temperature":
                0.5,


                "max_output_tokens":
                2500

            }

        )



    def generate_advice(
        self,
        prediction,
        weak_features
    ):


        prompt = f"""

You are an expert AI academic mentor.

Analyze the student's performance.

Student predicted score:
{round(prediction,2)}

Weak areas:
{weak_features}


Generate a COMPLETE detailed report:


## 📊 Performance Analysis

Explain current performance level.


## ⚠️ Weakness Explanation

Explain every weakness clearly.


## 🎯 Personalized Improvement Roadmap

Give:
- daily plan
- weekly plan
- priority areas


## 📚 Study Strategy

Give:
- learning methods
- practice strategy
- revision approach


## 🚀 Final Recommendation

Give motivational but practical advice.


Rules:
- Do not cut the answer
- Complete every section
- Use bullet points
- Be specific
- Avoid generic advice

"""


        response = self.model.generate_content(
            prompt
        )


        return response.text