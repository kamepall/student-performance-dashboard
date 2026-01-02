import streamlit as st
import pandas as pd
import plotly.express as px
import requests, zipfile
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import math

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="College Academic Portal",
    layout="wide"
)

# --------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------
defaults = {
    "logged_in": False,
    "role": None,
    "data": None,
    "model": None,
    "rmse": None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --------------------------------------------------
# LOGIN
# --------------------------------------------------
def login_ui():
    st.title("🔐 College Portal Login")
    role = st.selectbox("Login as", ["Student", "Faculty"])
    if st.button("Login"):
        st.session_state.logged_in = True
        st.session_state.role = role
        st.rerun()

if not st.session_state.logged_in:
    login_ui()
    st.stop()

# --------------------------------------------------
# LOGOUT
# --------------------------------------------------
with st.sidebar:
    st.success(f"Logged in as: {st.session_state.role}")
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    r = requests.get(url, timeout=10)
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        return pd.read_csv(z.open("student-mat.csv"), sep=";")

if st.session_state.data is None:
    with st.spinner("Loading academic data..."):
        st.session_state.data = load_data()

df = st.session_state.data.copy()

# --------------------------------------------------
# DERIVED COLUMNS
# --------------------------------------------------
df["Result"] = df["G3"].apply(lambda x: "Pass" if x >= 10 else "Fail")

# --------------------------------------------------
# FILTERS
# --------------------------------------------------
st.sidebar.header("🎛 Filters")

gender_filter = st.sidebar.multiselect(
    "Select Gender",
    options=df["sex"].unique(),
    default=list(df["sex"].unique())
)

grade_range = st.sidebar.slider(
    "Final Grade Range (G3)",
    min_value=int(df["G3"].min()),
    max_value=int(df["G3"].max()),
    value=(int(df["G3"].min()), int(df["G3"].max()))
)

filtered_df = df[
    (df["sex"].isin(gender_filter)) &
    (df["G3"].between(grade_range[0], grade_range[1]))
]

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📊 Student Dashboard", "👨‍🏫 Faculty Analytics", "📤 Export"]
)

# ==================================================
# TAB 1: STUDENT DASHBOARD
# ==================================================
with tab1:
    st.header("📊 Student Academic Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Final Grade", f"{filtered_df['G3'].mean():.2f}")
    col2.metric("Pass %", f"{(filtered_df['Result']=='Pass').mean()*100:.1f}%")
    col3.metric("Avg Absences", f"{filtered_df['absences'].mean():.1f}")

    st.plotly_chart(
        px.histogram(
            filtered_df,
            x="G3",
            nbins=20,
            title="Final Grade Distribution"
        ),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter(
            filtered_df,
            x="absences",
            y="G3",
            color="sex",
            title="Absences vs Final Grade"
        ),
        use_container_width=True
    )

    pf_df = filtered_df["Result"].value_counts().reset_index()
    pf_df.columns = ["Status", "Count"]

    st.plotly_chart(
        px.pie(
            pf_df,
            names="Status",
            values="Count",
            title="Pass vs Fail Ratio"
        ),
        use_container_width=True
    )

# ==================================================
# TAB 2: FACULTY ANALYTICS
# ==================================================
with tab2:
    st.header("👨‍🏫 Faculty Analytics & Predictive Model")

    if st.session_state.role != "Faculty":
        st.warning("Faculty access only")
        st.stop()

    st.plotly_chart(
        px.box(
            filtered_df,
            x="sex",
            y="G3",
            title="Gender vs Performance"
        ),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter(
            filtered_df,
            x="studytime",
            y="G3",
            trendline="ols",
            title="Study Time vs Final Grade"
        ),
        use_container_width=True
    )

    if st.button("Train & Evaluate Model"):
        model_df = pd.get_dummies(filtered_df, drop_first=True)
        X = model_df.drop("G3", axis=1)
        y = model_df["G3"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        with st.spinner("Training Random Forest..."):
            model = RandomForestRegressor(
                n_estimators=150,
                random_state=42
            )
            model.fit(X_train, y_train)

        rmse = math.sqrt(
            mean_squared_error(y_test, model.predict(X_test))
        )

        st.session_state.model = model
        st.session_state.rmse = rmse

        st.success(f"Model RMSE: {rmse:.2f}")

        fi = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).head(10)

        st.plotly_chart(
            px.bar(
                fi,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top Influencing Factors"
            ),
            use_container_width=True
        )

# ==================================================
# TAB 3: EXPORT (CSV + PDF)
# ==================================================
with tab3:
    st.header("📤 Export Data & Reports")

    # CSV EXPORT
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Filtered Data (CSV)",
        csv,
        "student_data.csv",
        "text/csv"
    )

    # PDF EXPORT
    def generate_pdf(data):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            doc = SimpleDocTemplate(tmp.name)
            styles = getSampleStyleSheet()
            content = [
                Paragraph("College Academic Report", styles["Title"]),
                Paragraph(f"Records: {len(data)}", styles["Normal"]),
                Paragraph(f"Average Grade: {data['G3'].mean():.2f}", styles["Normal"]),
                Paragraph(f"Pass Percentage: {(data['Result']=='Pass').mean()*100:.1f}%", styles["Normal"]),
            ]
            doc.build(content)
            return tmp.name

    if st.button("📄 Generate PDF Report"):
        pdf_path = generate_pdf(filtered_df)
        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇ Download PDF Report",
                f,
                "academic_report.pdf",
                "application/pdf"
            )
