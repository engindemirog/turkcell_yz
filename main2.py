import pandas as pd
import numpy as np
import joblib

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "days_since_last_order": np.random.randint(0, 365, n),
    "shipping_duration_days" : np.random.randint(1,15,n),
    "used_coupon": np.random.choice([0,1], size=n),
    "order_amount" : np.random.uniform(50,1000,n),
})

#hedef değişken
df["order_canceled"] = (
    (df["days_since_last_order"]>180) | ((df["used_coupon"]==1) & (df["shipping_duration_days"]>7))).astype(int)


from sklearn.preprocessing import StandardScaler

X = df[["days_since_last_order","shipping_duration_days","used_coupon","order_amount"]]
y = df["order_canceled"]

#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler,"scaler.pkl")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

models = [
    #("Logistic Regression", LogisticRegression()),
    ("Random Forest", RandomForestClassifier(random_state=42))
]

from sklearn.metrics import accuracy_score

results = []

for name, model in models:
    model.fit(X_scaled,y)

#modeli eğittikten sonra kaydet


joblib.dump(model,"random_forest_model.pkl")
#print("Model kaydedildi")

#pip install joblib

#GET,POST,PUT,DELETE...

from fastapi import FastAPI
from pydantic import BaseModel
#validation

app = FastAPI(title="Order Return Predictor")

model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
 
class OrderData(BaseModel):
    days_since_last_order:int
    shipping_duration_days:int
    used_coupon : int
    order_amount : float


# pasaj.com.tr/api/orders/predict --> endpoint
@app.post("/predict")
def predict_order_return(data:OrderData):
    input_df = pd.DataFrame([data.dict()])
    
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    
    

    return {
        "order_canceled": int(prediction)
    }




                  