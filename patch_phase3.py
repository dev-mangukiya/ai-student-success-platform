import re

with open("dashboard/app.py", "r") as f:
    content = f.read()

# 1. Update model paths
content = content.replace('REGRESSOR_MODEL_PATH = "src/artifacts/xgboost_regressor.pkl"', 'REGRESSOR_MODEL_PATH = "src/artifacts/advanced_regressor.pkl"')
content = content.replace('PREPROCESSOR_PATH = "src/artifacts/preprocessor.pkl"', 'PREPROCESSOR_PATH = "src/artifacts/advanced_preprocessor.pkl"')
content = content.replace('from src.config import REGRESSOR_MODEL_PATH, PREPROCESSOR_PATH', 'from src.config import REGRESSOR_MODEL_PATH, PREPROCESSOR_PATH\nfrom src.components.database import save_prediction, get_student_history')

# Fix paths if config doesn't export them correctly or if they were hardcoded differently.
content = content.replace('from src.components import rag_advisor', 'import src.components.database as db\nfrom src.components import rag_advisor')
# Wait, let's just make sure config paths are correct. The config.py might not have advanced_regressor.pkl
# I'll just change the import or redefine them in app.py
content = content.replace('from src.config import REGRESSOR_MODEL_PATH, PREPROCESSOR_PATH', 'import os\nBASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\nREGRESSOR_MODEL_PATH = os.path.join(BASE_DIR, "src", "artifacts", "advanced_regressor.pkl")\nPREPROCESSOR_PATH = os.path.join(BASE_DIR, "src", "artifacts", "advanced_preprocessor.pkl")')

# 2. Update Student Telemetry Sidebar
sidebar_orig_pattern = re.compile(r'    if app_mode == "Student Telemetry":(.*?)    else:', re.DOTALL)
match = sidebar_orig_pattern.search(content)

sidebar_new = """    if app_mode == "Student Telemetry":
        st.markdown("<p style='color: #64748B; font-size: 0.9rem; margin-bottom: 25px;'>Input individual parameters to generate telemetry.</p>", unsafe_allow_html=True)
        
        student_id = st.text_input("Student ID (For DB Tracking)", value="STU-001")
        
        with st.expander("👤 Demographics & Background", expanded=True):
            age_at_enrollment = st.slider("Age at Enrollment", 17, 60, 20)
            gender = st.selectbox("Gender", ["male", "female"])
            marital_status = st.selectbox("Marital Status", ["single", "married", "other"])
            displacement = st.selectbox("Displacement", ["yes", "no"])
            parent_education = st.selectbox("Parent Education", ["high_school", "bachelors", "masters", "phd", "none"])
            parent_occupation = st.selectbox("Parent Occupation", ["white_collar", "blue_collar", "unemployed", "business"])
            
        with st.expander("💰 Financials", expanded=False):
            tuition_fees = st.selectbox("Tuition Fees Up To Date", ["yes", "no"])
            scholarship = st.selectbox("Scholarship Holder", ["yes", "no"])
            debtor = st.selectbox("Debtor", ["yes", "no"])
            
        with st.expander("🌍 Macroeconomic Variables", expanded=False):
            inflation_rate = st.slider("Inflation Rate (%)", -2.0, 10.0, 2.5)
            gdp_growth = st.slider("GDP Growth (%)", -5.0, 10.0, 1.5)
            unemployment_rate = st.slider("Unemployment Rate (%)", 0.0, 25.0, 5.5)
            
        with st.expander("📚 Academic Behavior", expanded=False):
            s1_enrolled = st.slider("S1 Units Enrolled", 0, 10, 5)
            s1_approved = st.slider("S1 Units Approved", 0, 10, 5)
            s1_attendance = st.slider("S1 Seminar Attendance (%)", 0, 100, 85)
            
            s2_enrolled = st.slider("S2 Units Enrolled", 0, 10, 5)
            s2_approved = st.slider("S2 Units Approved", 0, 10, 5)
            s2_attendance = st.slider("S2 Seminar Attendance (%)", 0, 100, 85)

        st.markdown("<hr style='border: none; height: 1px; background: rgba(255,255,255,0.1); margin: 30px 0;'/>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #94A3B8; font-weight: 600; font-size: 1rem;'>True Document RAG</h4>", unsafe_allow_html=True)
        uploaded_syllabus = st.file_uploader("Upload Course Syllabus (PDF)", type=["pdf"], help="The AI will ingest this document and tailor your study plan to the specific course material.")
        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("Initialize Sequence 🚀", use_container_width=True, type="primary")
"""
if match:
    content = content.replace(match.group(0), sidebar_new + "    else:")


# 3. Update main prediction dataframe
main_df_orig_pattern = re.compile(r'        input_data = pd\.DataFrame\(\s*\{.*?\s*\}\s*\)', re.DOTALL)
match_df = main_df_orig_pattern.search(content)

main_df_new = """        input_data = pd.DataFrame(
            {
                "age_at_enrollment": [age_at_enrollment],
                "gender": [gender],
                "marital_status": [marital_status],
                "displacement": [displacement],
                "parent_education": [parent_education],
                "parent_occupation": [parent_occupation],
                "tuition_fees_up_to_date": [tuition_fees],
                "scholarship_holder": [scholarship],
                "debtor": [debtor],
                "inflation_rate": [inflation_rate],
                "gdp_growth": [gdp_growth],
                "unemployment_rate": [unemployment_rate],
                "s1_curricular_units_enrolled": [s1_enrolled],
                "s1_curricular_units_approved": [s1_approved],
                "s1_seminar_attendance": [s1_attendance],
                "s2_curricular_units_enrolled": [s2_enrolled],
                "s2_curricular_units_approved": [s2_approved],
                "s2_seminar_attendance": [s2_attendance],
            }
        )
        
        # Save to DB
        st.session_state.transformed_input = preprocessor.transform(input_data)
        pred = model.predict(st.session_state.transformed_input)[0]
        st.session_state.prediction = pred
        
        db.save_prediction(student_id, input_data.iloc[0].to_dict(), pred)
        
        st.session_state.avg_score = (s1_attendance + s2_attendance) / 2  # mock base
        st.session_state.weak_features = []
        if s2_approved < s2_enrolled:
            st.session_state.weak_features.append("Failed Curricular Units")
        if tuition_fees == "no" or debtor == "yes":
            st.session_state.weak_features.append("Financial Hardship")
        if s2_attendance < 70:
            st.session_state.weak_features.append("Low Seminar Attendance")
        st.session_state.student_id = student_id
"""
# We have to be careful with replacing. The original block calculates weak features too.
full_pred_block_pattern = re.compile(r'        input_data = pd\.DataFrame\(.*?st\.session_state\.ai_advice = None', re.DOTALL)
match_pred = full_pred_block_pattern.search(content)
if match_pred:
    content = content.replace(match_pred.group(0), main_df_new)

# 4. Update Tabs 
tabs_orig = '        dashboard, charts, explain, ai = st.tabs(\n            [\n                "⊞ Command Center",\n                "🌌 Neural Visuals",\n                "🧠 XAI Explainability",\n                "🤖 Gemini Strategist",\n            ]\n        )'
tabs_new = '        dashboard, charts, explain, ai, tracking = st.tabs(\n            [\n                "⊞ Command Center",\n                "🌌 Neural Visuals",\n                "🧠 XAI Explainability",\n                "🤖 Gemini Strategist",\n                "📈 DB Tracking",\n            ]\n        )'
content = content.replace(tabs_orig, tabs_new)

# 5. Add Tracking tab content right after the AI tab
ai_tab_end_pattern = re.compile(r'            st\.session_state\.messages\.append\(\{"role": "assistant", "content": response\}\)\n\n        st\.markdown\("</div>", unsafe_allow_html=True\)')
match_ai = ai_tab_end_pattern.search(content)

tracking_tab = """
    with tracking:
        st.markdown("<div class='animate-in' style='margin-top: 25px;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #F8FAFC; font-weight: 800; font-size: 1.6rem;'>Longitudinal Database Tracking</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94A3B8;'>Historical predicted scores fetched directly from the JSON database.</p>", unsafe_allow_html=True)
        
        history = db.get_student_history(st.session_state.student_id)
        if len(history) > 0:
            hist_df = pd.DataFrame(history)
            # Ensure timestamp is datetime for proper plotting
            hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
            
            line_chart = px.line(
                hist_df, x="timestamp", y="predicted_score", 
                markers=True, title=f"Trajectory for {st.session_state.student_id}",
                color_discrete_sequence=["#38BDF8"]
            )
            line_chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8"))
            st.plotly_chart(line_chart, use_container_width=True)
        else:
            st.info("No historical data found for this student.")
            
        st.markdown("</div>", unsafe_allow_html=True)
"""
if match_ai:
    content = content.replace(match_ai.group(0), match_ai.group(0) + tracking_tab)


# Save
with open("dashboard/app.py", "w") as f:
    f.write(content)

