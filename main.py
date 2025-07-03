import joblib
from fastapi import FastAPI
from schemas.order_data import OrderData
import pandas as pd

#validation

app = FastAPI(title="Order Return Predictor")

model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")


# pasaj.com.tr/api/orders/predict --> endpoint
@app.post("/predict")
def predict_order_return(data:OrderData):
    input_df = pd.DataFrame([data.dict()])
    
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    return {
        "order_canceled": int(prediction)
    }