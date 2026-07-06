import os

from dotenv import load_dotenv

import google.generativeai as genai


load_dotenv()


api_key = os.getenv(
    "GOOGLE_API_KEY"
)


print(
    "API KEY FOUND:",
    api_key is not None
)


genai.configure(
    api_key=api_key
)


model = genai.GenerativeModel(
    "gemini-2.0-flash-lite"
)


try:


    response = model.generate_content(
        "Say Gemini is working"
    )


    print(
        "✅ GEMINI WORKING"
    )


    print(
        response.text
    )


except Exception as error:


    print(
        "❌ GEMINI FAILED"
    )


    print(
        error
    )
