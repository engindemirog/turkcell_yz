import pandas as pd
import numpy as np

def generate_data(n=1000,seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
    "days_since_last_order": np.random.randint(0, 365, n),
    "shipping_duration_days" : np.random.randint(1,15,n),
    "used_coupon": np.random.choice([0,1], size=n),
    "order_amount" : np.random.uniform(50,1000,n),
    })

    #hedef değişken
    df["order_canceled"] = (
    (df["days_since_last_order"]>180) | ((df["used_coupon"]==1) & (df["shipping_duration_days"]>7))).astype(int)

    return df