# 🎓 AI Student Success Intelligence Platform

An end-to-end **Machine Learning + Explainable AI + Generative AI (RAG)** system that predicts a student's academic performance, explains *why*, classifies their risk level, and generates a personalized improvement plan through an AI advisor.

> **Status:** 🚧 Phase 1 of 8 complete — project scaffolding, logging, and exception handling. See [Roadmap](#roadmap) below.

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
# then fill in GEMINI_API_KEY (needed from Phase 6 onward)

# 5. Verify setup
python main.py
```

## Running Tests

```bash
pytest -v
```

---

## Roadmap

- [x] **Phase 1** — Project setup, folder structure, config, logger, custom exceptions
- [ ] **Phase 2** — Dataset sourcing, ingestion, validation
- [ ] **Phase 3** — Data transformation, model training, evaluation
- [ ] **Phase 4** — MLflow experiment tracking
- [ ] **Phase 5** — SHAP explainability
- [ ] **Phase 6** — Gemini + LangChain + FAISS RAG advisor
- [ ] **Phase 7** — Streamlit dashboard (Home, Prediction, Explainability, AI Advisor, Analytics)
- [ ] **Phase 8** — Docker + CI/CD + deployment

---

## Architecture (high level)

```
Data Source → Ingestion → Validation → Transformation → Feature Engineering
  → Train/Test Split → Model Training → Evaluation → MLflow Tracking
  → Save Best Model → Prediction Pipeline → Streamlit UI
                                     ↓
                         SHAP Explainability
                                     ↓
              Gemini + LangChain + FAISS RAG → AI Student Advisor
```

*(A rendered architecture diagram + dashboard screenshots will be added as later phases are completed.)*

---

## License

MIT
