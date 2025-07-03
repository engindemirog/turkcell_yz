from data.generate_data import generate_data
from preprocessing.preprocessing import process_data
from model.model import train_model
import joblib
import pandas as pd

df =  generate_data()

X = df[["days_since_last_order","shipping_duration_days","used_coupon","order_amount"]]
y = df["order_canceled"]

X_scaled, scaler = process_data(X)

joblib.dump(scaler,"scaler.pkl")

model =  train_model(X_scaled,y)

joblib.dump(model,"random_forest_model.pkl")