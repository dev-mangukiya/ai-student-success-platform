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
            print("⚠️ Model uninitialized due to missing key. Using static fallback.", flush=True)
            return self._get_static_advice(prediction, weak_features)

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
            print("❌ GEMINI CALL FAILED. Triggering local backup advisor...", flush=True)
            print(f"Error Details: {e}", flush=True)
            
            # Instead of returning a raw error block to the UI, we return the hardcoded backup advice
            return self._get_static_advice(prediction, weak_features)


    def _get_static_advice(self, prediction, weak_features):
        """Generates a structured fallback report when the live API is restricted."""
        
        # Turn the weak features list into readable text
        weak_areas_str = ", ".join(weak_features) if weak_features else "No critical weak areas detected"
        
        # Determine performance tier wording
        if prediction >= 80:
            performance_tier = "Excellent / High Performer"
            focus_strategy = "Deepen conceptual mastery, work on advanced problem sets, and eliminate minor careless mistakes."
        elif prediction >= 60:
            performance_tier = "Average / Moderate Performer"
            focus_strategy = "Target core conceptual gaps, revisit foundational textbook problems, and increase timed test practice."
        else:
            performance_tier = "Needs Immediate Attention"
            focus_strategy = "Break down materials into micro-topics, establish clear daily accountability, and seek active remedial tutoring."

        fallback_report = f"""
> 💡 *Note: The live AI engine is temporarily busy. This automated structural report has been dynamically compiled by your local mentor backup system.*

---

## 1. 📊 Performance Analysis
* **Current Evaluation Tier:** {performance_tier}
* **Projected Score Expectation:** **{round(prediction, 2)}%**
* Based on current input trends, the student possesses a stable baseline but maintains noticeable variance across key educational benchmarks. Actionable remediation is required to stabilize consistency.

## 2. 📉 Weakness Explanation
* **Flagged Targets:** `{weak_areas_str}`
* Failure to address these designated areas directly weakens overall grade structures over successive terms. Gaps in these modules compound quickly, dragging down performance during final evaluation weights.

## 3. 📚 Personalized Study Roadmap
* **Weeks 1–2 (Foundation Reset):** Isolate core conceptual topics matching your identified weak areas. Spend 45 minutes daily drilling structural formulas and terminology without notes.
* **Weeks 3–4 (Application Shift):** Transition toward practical workbook problems. Move from open-book solutions to unassisted workflows.
* **Weeks 5+ (Exam Condition Drills):** Complete past testing modules under strict timed limits to adapt processing speeds.

## 4. 🕒 Daily Routine
* **🌅 Morning (07:00 AM - 08:30 AM):** High-retention cognitive work. Review abstract theory notes or high-priority memorization items.
* **🌇 Afternoon (04:00 PM - 06:00 PM):** Active practice block. Solve analytical questions, trace code metrics, or process workbook variations.
* **🌃 Evening (08:30 PM - 09:30 PM):** System optimization. Audit errors committed during afternoon drills and map out tasks for the following morning.

## 5. 🎯 Score Improvement Strategy
* **Actionable Plan:** {focus_strategy}
* Log every incorrect sample problem inside a dedicated error tracking journal. Re-attempt every flagged mistake exactly 48 hours later to ensure retention.

## 6. 🚀 Motivation Advice
* A projected performance score is a reflection of current parameters, not an absolute cap on final capabilities. Incremental daily study loops outperform erratic midnight cram sessions every single time. Reset your execution steps tomorrow morning and stick to the tracking framework!
"""
        return fallback_report


