import pandas as pd
from prophet import Prophet
import pickle
import os

# Load and prepare data
df = pd.read_csv("D:/Projects/BUSINESS_DASHBOARD/data/sales_data.csv")
df = df.rename(columns={"Order Date": "ds", "Sales": "y"})

# Ensure datetime format
df['ds'] = pd.to_datetime(df['ds'])

# Fit the model
model = Prophet()
model.fit(df)

# Create model directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the model
with open("models/sales_model.pkl", "wb") as f:
    pickle.dump(model, f)

import matplotlib.pyplot as plt


# Create future dates (e.g., forecast next 90 days)
future = model.make_future_dataframe(periods=90)

# Make the forecast
forecast = model.predict(future)

# Show forecasted values
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

# Plot the forecast
model.plot(forecast)
plt.title("Sales Forecast for Next 90 Days")
plt.xlabel("Date")
plt.ylabel("Predicted Sales")
plt.show()
