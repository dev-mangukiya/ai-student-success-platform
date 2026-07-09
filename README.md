# 🎓 Enterprise AI Student Success Intelligence Platform

An end-to-end **Machine Learning + Explainable AI + Generative AI (RAG) + Persistent Database** system that predicts a student's academic performance, explains *why*, classifies their risk level, tracks their longitudinal trajectory, and generates a personalized improvement plan through an interactive AI mentor.

> **Status:** 🚀 **V2 Enterprise Upgrade Complete** — Advanced 20-feature XGBoost Pipeline, True Document RAG (PDF Syllabi), Educator Batch Triage, Longitudinal Database Tracking, and Premium Glassmorphism UI fully deployed.

---

## 🚀 Live Demo

Experience the completely overhauled AI Student Success Intelligence Platform:

🔗 **Live Application:**  
https://dev-ai-student-success-platform.streamlit.app

---

## 🌟 V2 Massive Architecture Upgrade

This project recently underwent a 3-Phase Enterprise Upgrade, introducing cutting-edge AI integration:

* **True Document RAG (PDF):** Upload course syllabi directly into the system. The Gemini AI reads the documents and tailors its academic strategy precisely to the uploaded material.
* **Interactive AI Mentor:** A real-time chat interface embedded in the dashboard, allowing students to ask follow-up questions to the Gemini Strategist about their predictive telemetry.
* **Educator Command Center:** A bulk-processing mode allowing teachers and administrators to upload CSVs of entire classrooms (50+ students). The XGBoost pipeline instantly processes the batch, generating a Trajectory Scatter Plot and isolating high-risk anomalies in a Triage Matrix.
* **Advanced Neural Pipeline:** Transitioned from a basic 8-parameter toy dataset to an advanced, highly-correlated 20-feature enterprise dataset covering Financials (Tuition, Scholarships), Macroeconomics (Inflation, GDP), and Academic Behavior (Seminar Attendance, Approved Units).
* **Longitudinal DB Tracking:** Integrated a persistent JSON database. The system now saves every prediction made per student, mapping out historical predicted trajectories over time in a dynamic line chart.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **ML Engine** | Python, Pandas, Scikit-Learn, **XGBoost (20-Feature Model)** |
| **Explainability** | SHAP (Neural Decision Transparency) |
| **Generative AI** | Gemini 2.0 API, LangChain, RAG |
| **Vector Engine** | PyPDF2, FAISS |
| **Database** | Persistent JSON / File-Based DB Tracking |
| **Frontend** | Streamlit + Plotly (Custom CSS, Dark Space Theme, Glassmorphism) |
| **Deployment** | Streamlit Community Cloud, GitHub CI/CD |

---

## 💻 Local Setup

```bash
# 1. Clone
git clone https://github.com/dev-mangukiya/ai-student-success-platform.git
cd ai-student-success-platform

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add Gemini API Key
# (Store this safely in your environment variables or a .env file)
export GEMINI_API_KEY="your_api_key_here"

# 5. Run Application
streamlit run dashboard/app.py
```

*(Note: If you want to use the Educator Command Center, you can generate a synthetic classroom dataset directly from the sidebar within the app!)*

---

## 🏛️ Architecture (High Level)

```
Advanced 20-Feature Data Source 
      ↓
Data Preprocessing (OneHot, StandardScale)
      ↓
XGBoost Regression Model
      ↓
Streamlit Executive Dashboard ──→ Persistent JSON Database (Longitudinal Tracking)
      ↓
SHAP Explainability (Waterfall Charts)
      ↓
Document RAG (PyPDF2) + Gemini Mentor Chat
```

---

## 👨‍💻 Author

### Dev Mangukiya

AI & Data Science Student | AI/ML Developer

Building intelligent, enterprise-grade AI systems using:
- Machine Learning & Deep Learning
- Generative AI & Retrieval-Augmented Generation (RAG)
- Explainable AI
- MLOps & Full-Stack Deployment

### 🌐 Connect With Me

🔗 **LinkedIn:**  
https://www.linkedin.com/in/devmangukiya/

🔗 **GitHub:**  
https://github.com/dev-mangukiya

---

## License

MIT
