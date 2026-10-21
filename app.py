import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Student AI Analytics",
    page_icon="🎓",
    layout="wide"
)

# Generate sample student data
@st.cache_data
def create_data():
    np.random.seed(42)
    n = 500

    hours = np.random.uniform(1, 8, n).round(1)
    attendance = np.random.randint(40, 101, n)
    assignment = np.random.randint(30, 101, n)
    previous = np.random.randint(25, 101, n)
    noise = np.random.normal(0, 5, n)

    marks = np.clip(
        hours * 4 + attendance * 0.20
        + assignment * 0.25 + previous * 0.25
        + noise, 0, 100
    ).round(1)

    return pd.DataFrame({
        "Study Hours": hours,
        "Attendance": attendance,
        "Assignment Score": assignment,
        "Previous Marks": previous,
        "Final Marks": marks
    })

df = create_data()

features = [
    "Study Hours", "Attendance",
    "Assignment Score", "Previous Marks"
]
model = RandomForestRegressor(
    n_estimators=100, random_state=42
)
model.fit(df[features], df["Final Marks"])

# Styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #111c35);
    color: white;
}
[data-testid="stMetric"] {
    background: #1b2942;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #304666;
}
h1, h2, h3 {
    color: #80c7ff;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Student AI Performance Analytics")
st.caption("Machine Learning • Academic Insights • Performance Prediction")
st.divider()

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Students", f"{len(df):,}")
c2.metric("Average Marks", f"{df['Final Marks'].mean():.1f}/100")
c3.metric("Average Attendance", f"{df['Attendance'].mean():.1f}%")
c4.metric("Pass Rate", f"{(df['Final Marks'] >= 50).mean()*100:.1f}%")

st.divider()
st.subheader("📊 Student Performance Overview")

left, right = st.columns(2)

with left:
    fig = px.histogram(
        df, x="Final Marks", nbins=20,
        title="Distribution of Final Marks",
        template="plotly_dark",
        color_discrete_sequence=["#52b6ff"]
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.scatter(
        df, x="Study Hours", y="Final Marks",
        color="Attendance",
        title="Study Hours vs Final Marks",
        template="plotly_dark",
        color_continuous_scale="blues"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("🤖 AI Student Marks Prediction")
st.write("Enter a student's details to estimate their final marks.")

a, b = st.columns(2)
with a:
    hours = st.slider("Daily study hours", 0.0, 12.0, 6.0, 0.5)
    attendance = st.slider("Attendance (%)", 0, 100, 85)
with b:
    assignment = st.slider("Assignment score", 0, 100, 80)
    previous = st.slider("Previous marks", 0, 100, 75)

if st.button("✨ Predict Student Performance", type="primary"):
    student = pd.DataFrame([{
        "Study Hours": hours,
        "Attendance": attendance,
        "Assignment Score": assignment,
        "Previous Marks": previous
    }])

    prediction = float(model.predict(student)[0])
    st.metric("Estimated Final Marks", f"{prediction:.1f}/100")

    if prediction >= 75:
        st.success("Predicted category: Excellent")
    elif prediction >= 50:
        st.info("Predicted category: Average")
    else:
        st.warning("Predicted category: Needs Improvement")

    if attendance < 75:
        st.write("💡 Recommendation: Improve class attendance.")
    if hours < 3:
        st.write("💡 Recommendation: Consider a regular study schedule.")
    if assignment < 50:
        st.write("💡 Recommendation: Focus on assignment practice.")

st.divider()
st.subheader("📋 Sample Student Records")
st.dataframe(df.head(15), use_container_width=True)

st.caption(
    "Educational demo using synthetic data. "
    "Predictions are not validated for real students."
)
