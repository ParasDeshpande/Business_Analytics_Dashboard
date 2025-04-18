from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Load models
churn_model = pickle.load(open("models/churn_model.pkl", "rb"))
sales_model = pickle.load(open("models/sales_model.pkl", "rb"))
sentiment_model, vectorizer = pickle.load(open("models/sentiment_model.pkl", "rb"))

# Input models
class ChurnInput(BaseModel):
    age: int
    gender: int
    MonthlyCharges: float

class SentimentInput(BaseModel):
    text: str

class ForecastInput(BaseModel):
    days: int = 30  # Default forecast horizon

@app.post("/predict-churn")
def predict_churn(data: ChurnInput):
    df = pd.DataFrame([data.dict()])
    pred = churn_model.predict(df)[0]
    return {"churn": int(pred)}

@app.post("/analyze-sentiment")
def analyze_sentiment(data: SentimentInput):
    vec = vectorizer.transform([data.text])
    pred = sentiment_model.predict(vec)[0]
    return {"sentiment": pred}

@app.post("/forecast-sales")
def forecast_sales(data: ForecastInput):
    future = sales_model.make_future_dataframe(periods=data.days)
    forecast = sales_model.predict(future)
    forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(data.days)

    # Convert to dict for JSON response
    result = forecast_data.to_dict(orient="records")
    return {"forecast": result}

# python -m uvicorn api.main:app --reload type in terminal
