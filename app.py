import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import requests, zipfile
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import math
pio.kaleido.scope.default_format = "png"


# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config("Student Performance Dashboard", layout="wide")

# ----------------------------------
# DARK MODE TOGGLE
# ----------------------------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:
    pio.templates.default = "plotly_dark"
else:
    pio.templates.default = "plotly"

# ----------------------------------
# LOAD DATA
# ----------------------------------
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    r = requests.get(url)
    r.raise_for_status()
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        df = pd.read_csv(z.open("student-mat.csv"), sep=";")
    return df

df = load_data()

# ----------------------------------
# MAIN TABS
# ----------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Model Comparison", "📄 Report"])

# =====================================================
# TAB 1: INTERACTIVE DASHBOARD
# =====================================================
with tab1:
    st.subheader("📊 Interactive Student Dashboard")

    fig = px.histogram(
        df,
        x="G3",
        title="Final Grade Distribution",
        nbins=20
    )

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(label="All",
                         method="update",
                         args=[{"visible": [True]}]),
                    dict(label="Pass (>=10)",
                         method="restyle",
                         args=[{"x": [df[df["G3"] >= 10]["G3"]]}]),
                    dict(label="Fail (<10)",
                         method="restyle",
                         args=[{"x": [df[df["G3"] < 10]["G3"]]}]),
                ],
                direction="down",
                showactive=True,
            )
        ]
    )

    st.plotly_chart(fig, use_container_width=True)

    scatter = px.scatter(
        df,
        x="absences",
        y="G3",
        color="sex",
        title="Absences vs Final Grade"
    )
    st.plotly_chart(scatter, use_container_width=True)

# =====================================================
# TAB 2: MODEL COMPARISON
# =====================================================
with tab2:
    st.subheader("🤖 Model Performance Comparison")

    model_df = pd.get_dummies(df, drop_first=True)
    X = model_df.drop("G3", axis=1)
    y = model_df["G3"]

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)

    rf.fit(X, y)
    xgb.fit(X, y)

    rf_rmse = math.sqrt(mean_squared_error(y, rf.predict(X)))
    xgb_rmse = math.sqrt(mean_squared_error(y, xgb.predict(X)))

    st.metric("Random Forest RMSE", f"{rf_rmse:.2f}")
    st.metric("XGBoost RMSE", f"{xgb_rmse:.2f}")

    best_model = rf if rf_rmse < xgb_rmse else xgb
    st.success(f"🏆 Best Model: {'Random Forest' if best_model == rf else 'XGBoost'}")

# =====================================================
# TAB 3: PDF REPORT WITH CHART
# =====================================================
with tab3:
    st.subheader("📄 Generate Performance Report")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        fig.write_image(tmp_img.name)

    def generate_pdf():
        pdf_file = "student_report.pdf"
        doc = SimpleDocTemplate(pdf_file)
        styles = getSampleStyleSheet()

        content = [
            Paragraph("Student Performance Report", styles["Title"]),
            Paragraph(f"Random Forest RMSE: {rf_rmse:.2f}", styles["Normal"]),
            Paragraph(f"XGBoost RMSE: {xgb_rmse:.2f}", styles["Normal"]),
            Image(tmp_img.name, width=400, height=300)
        ]

        doc.build(content)
        return pdf_file

    if st.button("📄 Generate PDF"):
        file = generate_pdf()
        with open(file, "rb") as f:
            st.download_button(
                "⬇ Download Report",
                f,
                file_name="student_performance_report.pdf",
                mime="application/pdf"
            )
