import streamlit as st
import requests
import pandas as pd

st.title("📊 AI-Powered Business Analytics Dashboard")
st.sidebar.title("🔍 Choose Analysis")

# Add Forecasting to the sidebar options
choice = st.sidebar.radio("Select", ["Churn Prediction", "Sentiment Analysis", "Sales Forecasting"])

# ----------------------------
# Churn Prediction
# ----------------------------
if choice == "Churn Prediction":
    st.subheader("🧍 Customer Churn Prediction")
    age = st.number_input("Age", 18, 100)
    gender = st.selectbox("Gender", ["Male", "Female"])
    charges = st.number_input("Monthly Charges", 0.0, 500.0)

    if st.button("Predict"):
        payload = {
            "Age": age,
            "Gender": 0 if gender == "Male" else 1,
            "MonthlyCharges": charges
        }
        response = requests.post("http://localhost:8000/predict-churn", json=payload)
        if response.ok:
            st.write("Prediction:", "Churn" if response.json()["churn"] == 1 else "No Churn")
        else:
            st.error("API error: could not get prediction.")

# ----------------------------
# Sentiment Analysis
# ----------------------------
if choice == "Sentiment Analysis":
    st.subheader("💬 Customer Review Sentiment")
    review = st.text_area("Enter Review Text")

    if st.button("Analyze"):
        response = requests.post("http://localhost:8000/analyze-sentiment", json={"text": review})
        if response.ok:
            st.write("Sentiment:", response.json()["sentiment"])
        else:
            st.error("API error: could not get sentiment analysis.")

# ----------------------------
# Sales Forecasting
# ----------------------------
if choice == "Sales Forecasting":
    st.subheader("📈 Sales Forecasting with Prophet")
    st.write(
        "🔹 Choose how many days into the future you'd like to forecast sales. "
        "You'll see predicted daily sales along with a line chart to visualize the trend.\n\n"
        "📘 **Note:** Higher predicted values (`yhat`) indicate expected growth. Use this to prepare inventory and marketing plans accordingly."
    )

    days = st.slider("Forecast Days", min_value=7, max_value=90, value=30, step=1)

    if st.button("Generate Forecast"):
        response = requests.post("http://localhost:8000/forecast-sales", json={"days": days})
        if response.ok:
            data = response.json()["forecast"]
            df = pd.DataFrame(data)
            st.write("📅 Forecasted Sales for Next", days, "Days:")
            st.dataframe(df)

            st.line_chart(df.set_index("ds")[["yhat"]])
            st.info(
                "✅ Tip: Use the trend to align your planning. Rising sales? Scale up! "
                "Dipping sales? Time for promotions or inventory review."
            )
        else:
            st.error("API error: could not generate forecast.")

# python -m streamlit run app.py