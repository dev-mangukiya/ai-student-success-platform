# 🎓 AI Student Success Intelligence Platform

An end-to-end **Machine Learning + Explainable AI + Generative AI (RAG)** system that predicts a student's academic performance, explains *why*, classifies their risk level, and generates a personalized improvement plan through an AI advisor.

> **Status:** 🚀 Production AI Platform — ML Pipeline, Explainable AI, Gemini Advisor, Premium Dashboard, and Cloud Deployment completed. See [Roadmap](#roadmap) below.

---

## 🚀 Live Demo

Experience the AI Student Success Intelligence Platform:

🔗 **Live Application:**  
https://dev-ai-student-success-platform.streamlit.app

---

## Tech Stack

| Layer | Tools |
|---|---|
| ML | Python, Pandas, NumPy, Scikit-Learn, XGBoost |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Generative AI | Gemini API, LangChain, RAG |
| Vector Store | FAISS |
| Frontend | Streamlit + Plotly (custom CSS, dark SaaS theme) |
| Deployment | Streamlit Community Cloud, Docker |
| CI/CD | GitHub Actions |

---

## Project Structure

```
student-success-ai/
├── data/
│   ├── raw/                  # original dataset
│   └── processed/            # train/test splits after transformation
├── artifacts/
│   ├── models/                # saved regressor/classifier .pkl
│   ├── preprocessors/         # saved encoders/scalers
│   └── reports/               # evaluation metrics, SHAP outputs
├── src/
│   ├── config.py               # single source of truth for paths & constants
│   ├── components/             # ingestion, validation, transformation, training, evaluation
│   ├── pipeline/                # training_pipeline.py, prediction_pipeline.py
│   └── utils/                  # logger.py, exception.py, helpers.py
├── ai/
│   └── rag/                    # vector_store.py, retriever.py, advisor.py, prompts.py
├── dashboard/
│   ├── Home.py
│   └── pages/                  # Prediction, Explainability, AI_Advisor, Analytics
├── notebooks/                 # EDA / experimentation
├── tests/                      # pytest suite
├── .github/workflows/          # CI pipeline
├── Dockerfile
├── requirements.txt
└── main.py                     # CLI entry point
```

---

## Local Setup

```bash
# 1. Clone
git clone <your-repo-url>
cd student-success-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env

# Add Gemini API Key
GEMINI_API_KEY=your_api_key_here

# 5. Run Application
streamlit run dashboard/app.py
```

---

## Running Tests

```bash
pytest -v
```

---

## Roadmap

- [x] **Phase 1** — Project setup, folder structure, config, logger, custom exceptions
- [x] **Phase 2** — Dataset sourcing, ingestion, validation
- [x] **Phase 3** — Data transformation, model training, evaluation
- [x] **Phase 4** — MLflow experiment tracking
- [x] **Phase 5** — SHAP explainability
- [x] **Phase 6** — Gemini + LangChain + FAISS RAG advisor
- [x] **Phase 7** — Streamlit dashboard (Home, Prediction, Explainability, AI Advisor, Analytics)
- [ ] **Phase 8** — Docker + CI/CD + production improvements

---

## Architecture (high level)

```
Data Source 
      ↓
Ingestion
      ↓
Validation
      ↓
Transformation
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Evaluation + MLflow Tracking
      ↓
Best Model Selection
      ↓
Prediction Pipeline
      ↓
Streamlit Dashboard

        ↓

SHAP Explainability

        ↓

Gemini + RAG AI Student Advisor
```

---

## 👨‍💻 Author

### Dev Mangukiya

AI & Data Science Student | AI/ML Developer

Building intelligent AI systems using:

- Machine Learning
- Deep Learning
- Generative AI
- Explainable AI
- MLOps


### 🌐 Connect With Me

🔗 **LinkedIn:**  
https://www.linkedin.com/in/devmangukiya/


---

## License

MIT

