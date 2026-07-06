import os

from dotenv import load_dotenv

import google.generativeai as genai



# ==========================
# LOAD ENV VARIABLES
# ==========================

load_dotenv()



# ==========================
# RAG ADVISOR
# ==========================

class RAGAdvisor:


    def __init__(
        self
    ):


        api_key = os.getenv(
            "GOOGLE_API_KEY"
        )


        print(
            "GEMINI API FOUND:",
            api_key is not None
        )


        if api_key is None:

            raise Exception(
                "GOOGLE_API_KEY missing"
            )



        genai.configure(
            api_key=api_key
        )


        self.model = genai.GenerativeModel(

            model_name=
            "gemini-2.0-flash-lite",

            generation_config={

                "temperature":
                0.7,


                "max_output_tokens":
                3000

            }

        )



    # ==========================
    # GENERATE ADVICE
    # ==========================

    def generate_advice(
        self,
        prediction,
        weak_features
    ):


        prompt = f"""


You are an expert AI Student Success Advisor.


Analyze this student's academic profile.


Student Predicted Score:

{round(prediction,2)}%


Weak Areas:

{weak_features}



Create a complete personalized report:


## 📊 Performance Analysis

Explain the student's current level.


## ⚠️ Weak Area Explanation

Explain each weak area.


## 🎯 Improvement Roadmap

Give:

- Daily plan
- Weekly plan
- Priority topics


## 📚 Study Strategy

Provide:

- Learning techniques
- Practice strategy
- Revision plan


## 🚀 Final Recommendation

Give practical improvement advice.


Rules:

- Complete every section
- Do not stop halfway
- Be specific
- Use markdown formatting


"""



        try:


            print(
                "Calling Gemini API..."
            )


            response = self.model.generate_content(
                prompt
            )


            print(
                "Gemini response received successfully"
            )


            return response.text



        except Exception as error:


            print(
                "Gemini failed because:",
                error
            )


            weak_text = (

                ", ".join(
                    weak_features
                )

                if weak_features

                else

                "No major weaknesses detected 🎉"

            )



            return f"""


## ⚠️ Gemini Offline Mode


(Gemini API failed. Using backup advisor)


---


## 📊 Performance Analysis


Predicted Performance:

### {round(prediction,2)}%


Your ML model prediction was generated successfully.



---


## ⚠️ Weak Areas


{weak_text}



---


## 🎯 Improvement Plan


- Focus on weak subjects first

- Create daily practice targets

- Review mistakes weekly

- Track progress regularly



---


## 📚 Study Strategy


- Study in focused sessions

- Practice previous problems

- Improve weak concepts

- Take regular assessments



---


## 🚀 Final Advice


Your AI/ML prediction pipeline is working.

Only Gemini response generation is unavailable temporarily.

"""