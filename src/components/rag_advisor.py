import os
import google.generativeai as genai

from dotenv import load_dotenv


load_dotenv()



class RAGAdvisor:


    def __init__(self):


        api_key = os.getenv(
            "GOOGLE_API_KEY"
        )


        if api_key is None:

            raise Exception(
                "GOOGLE_API_KEY not found in .env file"
            )


        genai.configure(
            api_key=api_key
        )


        self.model = genai.GenerativeModel(
            "gemini-2.5-flash-lite"
        )



    def generate_advice(
        self,
        prediction,
        weak_features
    ):


        prompt = f"""

        You are an AI student performance advisor.

        Student predicted score:
        {prediction}

        Weak areas detected:
        {weak_features}


        Provide:

        1. Performance analysis
        2. Weakness explanation
        3. Personalized improvement plan
        4. Study strategy


        Keep response:
        - practical
        - short
        - student friendly

        """



        try:


            response = self.model.generate_content(

                prompt,

                generation_config={

                    "max_output_tokens": 250,

                    "temperature": 0.3

                }

            )


            return response.text



        except Exception as e:


            return f"""

            AI Advisor temporarily unavailable.

            Reason:
            {e}


            ML prediction completed successfully.

            Predicted Score:
            {prediction}


            Weak Areas:
            {weak_features}

            """