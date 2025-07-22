#KMeans

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



np.random.seed(42)
n=300

income = np.random.randint(50000,150000,n)
spending = np.random.randint(500,20000,n)

df = pd.DataFrame({
    "annual_income" : income,
    "monthly_spending": spending
})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=3, random_state=42)

df["cluster"] = kmeans.fit_predict(X_scaled)

new_customer = pd.DataFrame({
    "annual_income" : [132000],
    "monthly_spending" : [7000]
})

new_customer_scaled = scaler.transform(new_customer)
predicted_cluster = kmeans.predict(new_customer_scaled)

print("Küme numarası : ",predicted_cluster[0])


import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

for label in df["cluster"].unique():
    plt.scatter(df[df["cluster"]==label]["annual_income"],
                df[df["cluster"]==label]["monthly_spending"],
                label = f"Cluster {label}"
                )

plt.xlabel("Yıllık Gelir")
plt.ylabel("Aylık Harcama")
plt.title("KMeans ile Müşteri Segmentasyonu")
plt.legend()
plt.grid(True)
plt.show()