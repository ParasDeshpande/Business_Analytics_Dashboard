import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load data
df = pd.read_csv("D:/Projects/BUSINESS_DASHBOARD/data/telco_churn.csv")

# Encode gender (assumes values are 'Male' and 'Female')
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# Define features and target
X = df[['age', 'gender', 'MonthlyCharges']]
y = df['Churn']

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Ensure 'models' directory exists
os.makedirs("models", exist_ok=True)

# Save model
with open("models/churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

