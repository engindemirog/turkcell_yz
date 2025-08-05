import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics import accuracy_score

app = FastAPI()

df = pd.read_csv("IMDBDataset.csv")

X = df["review"]
y = df["sentiment"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf,y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test,y_pred)

def analyze_sentiment(review:str)->str:
    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)
    return prediction[0]


class ReviewObject(BaseModel):
    review:str

@app.post("/analyze")
def analyze_review(reviewObject:ReviewObject):
    sentiment = analyze_sentiment(reviewObject.review)
    return {"review":reviewObject.review,"sentiment":sentiment}


