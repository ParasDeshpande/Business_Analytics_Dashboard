import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

df = pd.read_csv(r"D:/Projects/BUSINESS_DASHBOARD/data/reviews.csv")
X = df["Cleaned_Text"]
y = df["Sentiment"]

vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"))
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

with open("models/sentiment_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

# Sample review
sample_text = ["this product is amazing and I love it"]
sample_vec = vectorizer.transform(sample_text)

# Predict sentiment
prediction = model.predict(sample_vec)
print("Sentiment:", prediction[0])
