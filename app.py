import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests, zipfile
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config("Student Performance Dashboard", layout="wide")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    r = requests.get(url)
    r.raise_for_status()
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        df = pd.read_csv(z.open("student-mat.csv"), sep=";")
    return df

df = load_data()

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.title("🎛 Filters")

gender = st.sidebar.selectbox("Gender", ["All", "M", "F"])
internet = st.sidebar.selectbox("Internet Access", ["All", "yes", "no"])

filtered_df = df.copy()
if gender != "All":
    filtered_df = filtered_df[filtered_df["sex"] == gender]
if internet != "All":
    filtered_df = filtered_df[filtered_df["internet"] == internet]

# -------------------------------------------------
# MAIN TABS
# -------------------------------------------------
tab_overview, tab_analysis, tab_predict = st.tabs(
    ["📘 Overview", "📊 Analysis", "🤖 Prediction"]
)

# =================================================
# TAB 1: OVERVIEW (TABS INSIDE TABS)
# =================================================
with tab_overview:
    sub1, sub2 = st.tabs(["📄 Dataset", "📈 Summary"])

    with sub1:
        st.dataframe(filtered_df.head())
        st.download_button(
            "⬇ Download Filtered CSV",
            filtered_df.to_csv(index=False),
            "filtered_students.csv",
            "text/csv"
        )

    with sub2:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average Grade", f"{filtered_df['G3'].mean():.2f}")
        c2.metric("Max Grade", filtered_df["G3"].max())
        c3.metric("Min Grade", filtered_df["G3"].min())
        c4.metric("Pass Rate (%)", f"{(filtered_df['G3']>=10).mean()*100:.1f}")

# =================================================
# TAB 2: ANALYSIS (INTERACTIVE PLOTLY)
# =================================================
with tab_analysis:
    st.subheader("Interactive Visual Analytics")

    suba, subb = st.tabs(["📊 Distributions", "🔗 Relationships"])

    with suba:
        fig = px.histogram(
            filtered_df, x="G3",
            nbins=20,
            title="Final Grade Distribution",
            hover_data=["sex", "studytime"]
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.box(
            filtered_df,
            x="studytime",
            y="G3",
            title="Study Time vs Final Grade"
        )
        st.plotly_chart(fig, use_container_width=True)

    with subb:
        fig = px.scatter(
            filtered_df,
            x="absences",
            y="G3",
            color="sex",
            title="Absences vs Final Grade",
            hover_data=["studytime"]
        )
        st.plotly_chart(fig, use_container_width=True)

        corr = filtered_df.corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=True,
            title="Correlation Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)

# =================================================
# TAB 3: PREDICTION + PDF REPORT
# =================================================
with tab_predict:
    st.subheader("🎯 Predict Student Final Grade")

    model_df = pd.get_dummies(df, drop_first=True)
    X = model_df.drop("G3", axis=1)
    y = model_df["G3"]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    G1 = st.slider("Grade 1 (G1)", 0, 20, 10)
    G2 = st.slider("Grade 2 (G2)", 0, 20, 10)
    studytime = st.slider("Study Time", 1, 4, 2)
    absences = st.slider("Absences", 0, 50, 5)

    input_data = np.zeros(len(X.columns))
    for col in X.columns:
        if col == "G1": input_data[X.columns.get_loc(col)] = G1
        if col == "G2": input_data[X.columns.get_loc(col)] = G2
        if col == "studytime": input_data[X.columns.get_loc(col)] = studytime
        if col == "absences": input_data[X.columns.get_loc(col)] = absences

    prediction = model.predict([input_data])[0]
    st.success(f"Predicted Final Grade: {prediction:.2f}")

    # ---------------- PDF REPORT ----------------
    def generate_pdf():
        file = "student_report.pdf"
        doc = SimpleDocTemplate(file)
        styles = getSampleStyleSheet()

        content = [
            Paragraph("Student Performance Report", styles["Title"]),
            Paragraph(f"Predicted Final Grade: {prediction:.2f}", styles["Normal"]),
            Paragraph("Input Summary:", styles["Heading2"]),
            Table([
                ["G1", G1],
                ["G2", G2],
                ["Study Time", studytime],
                ["Absences", absences]
            ])
        ]

        doc.build(content)
        return file

    if st.button("🧾 Generate PDF Report"):
        pdf_file = generate_pdf()
        with open(pdf_file, "rb") as f:
            st.download_button(
                "⬇ Download Report",
                f,
                file_name="student_report.pdf",
                mime="application/pdf"
            )
