import re

with open("dashboard/app.py", "r") as f:
    content = f.read()

# 1. Add session state
state_code = """if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None
if "batch_triggered" not in st.session_state:
    st.session_state.batch_triggered = False
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Student Telemetry"

@st.cache_data
def generate_synthetic_data():
    import numpy as np
    np.random.seed(42)
    n = 65
    data = pd.DataFrame({
        "gender": np.random.choice(["male", "female"], n),
        "race/ethnicity": np.random.choice(["group A", "group B", "group C", "group D", "group E"], n),
        "parental level of education": np.random.choice(["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"], n),
        "lunch": np.random.choice(["standard", "free/reduced"], n),
        "test preparation course": np.random.choice(["completed", "none"], n),
        "math score": np.random.randint(30, 100, n),
        "reading score": np.random.randint(30, 100, n),
        "writing score": np.random.randint(30, 100, n)
    })
    return data.to_csv(index=False).encode('utf-8')
"""
content = content.replace('if "pdf_text" not in st.session_state:\n    st.session_state.pdf_text = None', state_code)

# 2. Modify sidebar
sidebar_orig = """with st.sidebar:
    st.markdown(
        "<h2 style='color: #F8FAFC; font-weight: 800; font-size: 1.5rem; margin-bottom: 5px;'>System Config</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color: #64748B; font-size: 0.9rem; margin-bottom: 25px;'>Input student parameters to generate predictive telemetry.</p>",
        unsafe_allow_html=True,
    )"""

sidebar_new = """with st.sidebar:
    st.markdown(
        "<h2 style='color: #F8FAFC; font-weight: 800; font-size: 1.5rem; margin-bottom: 5px;'>System Config</h2>",
        unsafe_allow_html=True,
    )
    
    app_mode = st.radio("Operating Mode", ["Student Telemetry", "Educator Command Center"], label_visibility="collapsed")
    st.markdown("<hr style='border: none; height: 1px; background: rgba(255,255,255,0.1); margin: 15px 0;'/>", unsafe_allow_html=True)
    st.session_state.app_mode = app_mode

    if app_mode == "Student Telemetry":
        st.markdown("<p style='color: #64748B; font-size: 0.9rem; margin-bottom: 25px;'>Input individual parameters to generate telemetry.</p>", unsafe_allow_html=True)"""
content = content.replace(sidebar_orig, sidebar_new)

# 3. Indent the rest of the sidebar
sidebar_rest_pattern = re.compile(r'(    st\.markdown\(\s*"<h4 style=\'color: #94A3B8; font-weight: 600; font-size: 1rem; margin-top: 10px;\'>Demographics</h4>".*?    predict_clicked = st\.button\(\s*"Initialize Sequence 🚀", use_container_width=True, type="primary"\s*\))', re.DOTALL)
match = sidebar_rest_pattern.search(content)
if match:
    orig_block = match.group(1)
    indented_block = "\n".join(["    " + line if line.strip() else line for line in orig_block.split("\n")])
    
    educator_sidebar = """
    else:
        st.markdown("<p style='color: #64748B; font-size: 0.9rem; margin-bottom: 25px;'>Upload class data for bulk predictive triage.</p>", unsafe_allow_html=True)
        uploaded_csv = st.file_uploader("Upload Classroom CSV", type=["csv"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Sample CSV",
            data=generate_synthetic_data(),
            file_name="sample_classroom.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        batch_predict_clicked = st.button("Initialize Batch Analysis 🚀", use_container_width=True, type="primary")
        predict_clicked = False # mock to prevent error
"""
    content = content.replace(orig_block, indented_block + educator_sidebar)


# 4. Main UI split
main_ui_orig = """if predict_clicked:"""
main_ui_new = """if st.session_state.app_mode == "Student Telemetry":
    if predict_clicked:"""
content = content.replace(main_ui_orig, main_ui_new)

# Indent the entire existing block from if predict_clicked to the end of the file
# We'll split the file at the main_ui_new, and indent everything after it
parts = content.split('    if predict_clicked:')
if len(parts) == 2:
    tail = parts[1]
    indented_tail = "\n".join(["    " + line if line.strip() else line for line in tail.split("\n")])
    
    educator_main = """
else:
    if batch_predict_clicked:
        if uploaded_csv is not None:
            st.session_state.batch_triggered = True
            batch_df = pd.read_csv(uploaded_csv)
            # Transform
            transformed_batch = preprocessor.transform(batch_df)
            preds = model.predict(transformed_batch)
            batch_df["Predicted Score"] = preds
            batch_df["Base Avg"] = batch_df[["math score", "reading score", "writing score"]].mean(axis=1)
            batch_df["Risk Level"] = batch_df["Predicted Score"].apply(lambda x: "High Risk" if x < 65 else ("Medium" if x < 75 else "Optimal"))
            st.session_state.batch_results = batch_df
        else:
            st.error("Please upload a CSV file first.")
            
    if st.session_state.batch_triggered and st.session_state.batch_results is not None:
        batch_df = st.session_state.batch_results
        
        st.markdown("<div class='animate-in' style='margin-top: 25px;'>", unsafe_allow_html=True)
        st.markdown("<h2 style='color: #F8FAFC; font-weight: 800; font-size: 2.2rem;'>Educator Command Center</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        high_risk_count = len(batch_df[batch_df["Risk Level"] == "High Risk"])
        with col1:
            st.markdown(f"<div class='glass-card'><div class='metric-label'>TOTAL STUDENTS</div><div class='metric-value gradient-text'>{len(batch_df)}</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='glass-card'><div class='metric-label'>CLASS AVERAGE (PREDICTED)</div><div class='metric-value'>{batch_df['Predicted Score'].mean():.1f}</div></div>", unsafe_allow_html=True)
        with col3:
            glow = "rgba(244,63,94,0.3)" if high_risk_count > 0 else "rgba(16,185,129,0.3)"
            color = "#F43F5E" if high_risk_count > 0 else "#10B981"
            st.markdown(f"<div class='glass-card' style='box-shadow: 0 10px 30px {glow};'><div class='metric-label' style='color:{color}'>HIGH-RISK ANOMALIES</div><div class='metric-value' style='color:{color}; -webkit-text-fill-color: {color};'>{high_risk_count}</div></div>", unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        chart_col, table_col = st.columns([1.5, 1])
        with chart_col:
            st.markdown("<h4 style='color: #F8FAFC;'>Classroom Trajectory Scatter</h4>", unsafe_allow_html=True)
            scatter = px.scatter(
                batch_df, x="Base Avg", y="Predicted Score", color="Risk Level",
                color_discrete_map={"High Risk": "#F43F5E", "Medium": "#FACC15", "Optimal": "#10B981"},
                hover_data=["math score", "reading score", "writing score"]
            )
            scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8"))
            st.plotly_chart(scatter, use_container_width=True)
            
        with table_col:
            st.markdown("<h4 style='color: #F43F5E;'>Triage Matrix (High Risk)</h4>", unsafe_allow_html=True)
            high_risk_df = batch_df[batch_df["Risk Level"] == "High Risk"][["Predicted Score", "math score", "reading score", "writing score"]]
            if len(high_risk_df) > 0:
                st.dataframe(high_risk_df.style.background_gradient(cmap='Reds_r', subset=['Predicted Score']), use_container_width=True, height=400)
            else:
                st.success("No high-risk students detected!")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            \"\"\"
            <div class='animate-in' style='text-align: center; padding: 120px 40px; margin-top: 40px;'>
                <div style='font-size: 5rem; margin-bottom: 25px; animation: pulse 3s infinite; text-shadow: 0 0 30px rgba(129, 140, 248, 0.5);'>📊</div>
                <h3 style='color: #F8FAFC; font-size: 2.2rem; font-family: "Outfit", sans-serif; font-weight: 800; margin: 0 0 15px 0; letter-spacing: 1px;'>AWAITING CLASSROOM DATA</h3>
                <p style='color: #64748B; max-width: 600px; margin: 0 auto; font-size: 1.2rem; line-height: 1.6;'>
                    Upload a CSV file in the sidebar to process an entire classroom simultaneously. Use the Download Sample button if you need a template.
                </p>
            </div>
            \"\"\",
            unsafe_allow_html=True,
        )
"""
    content = parts[0] + '    if predict_clicked:' + indented_tail + educator_main

with open("dashboard/app.py", "w") as f:
    f.write(content)

