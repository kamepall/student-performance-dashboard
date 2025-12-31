# 🎓 Student Performance Analysis Dashboard

An end-to-end **Data Science & Machine Learning dashboard** that analyzes student academic performance, provides rich interactive visualizations, predicts final exam scores, and generates downloadable PDF reports — all deployed using **Streamlit Cloud**.

---

## 🚀 Live Application
🔗 **Streamlit App:** _(Add your deployed Streamlit URL here)_

---

## 📌 Project Overview

This project uses the **UCI Student Performance Dataset** to:
- Explore factors affecting student grades
- Perform comprehensive exploratory data analysis (EDA)
- Build a **Random Forest regression model** to predict final grades
- Provide an **interactive dashboard** with filters, charts, metrics, and reports

The dataset is fetched dynamically from the **UCI Machine Learning Repository** (no local storage required).

---

## 🧠 Features

### 📊 Data Analysis & Visualization
- Interactive **Plotly** charts (hover, zoom, filter)
- Histograms, box plots, scatter plots, correlation heatmaps
- Gender, internet access, and study time filters
- Metrics: average grade, max/min, pass rate

### 🤖 Machine Learning
- Random Forest Regressor
- Predicts final student grade (G3)
- Real-time predictions using user inputs

### 🎨 Advanced UI
- Tabs inside tabs (multi-level navigation)
- Clean, responsive Streamlit layout
- Sidebar filtering

### 🧾 Report Generation
- One-click **PDF report generation**
- Includes prediction results and input summary
- Downloadable directly from the app

---

## 🗂 Dataset Information

- **Source:** UCI Machine Learning Repository  
- **Dataset:** Student Performance (Math)  
- **Target Variable:** `G3` (Final Grade)

The dataset is accessed via a public URL and loaded at runtime.

---

## 🛠 Tech Stack

| Category | Tools |
|-------|------|
| Language | Python |
| Dashboard | Streamlit |
| Data Analysis | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Machine Learning | Scikit-learn (Random Forest) |
| Reporting | ReportLab |
| Deployment | Streamlit Cloud |
| Version Control | GitHub |

---

## 📁 Project Structure

