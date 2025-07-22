#Kmeans

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

np.random.seed(42)

def find_elbow(inertia_list):
    n_points = len(inertia_list)
    all_k = np.arange(1, n_points + 1)
    
    # İlk ve son noktayı referans alarak bir doğru çizer
    point1 = np.array([1, inertia_list[0]])
    point2 = np.array([n_points, inertia_list[-1]])
    
    # Doğrunun vektörü
    line_vec = point2 - point1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    
    # Her noktanın bu doğruya uzaklığını hesaplar
    distances = []
    for i in range(n_points):
        point = np.array([all_k[i], inertia_list[i]])
        vec_from_line = point - point1
        # Dikey uzaklık vektörü (projeksiyon farkı)
        dist = np.linalg.norm(vec_from_line - np.dot(vec_from_line, line_vec_norm) * line_vec_norm)
        distances.append(dist)

    optimal_k = distances.index(max(distances)) + 1  # çünkü k'ler 1'den başlıyor
    return optimal_k

n = 10000

data = pd.DataFrame({
    "annual_spend" : np.random.normal(500,200,n).clip(50,2000),
    "visit_frequency" : np.random.normal(10,5,n).clip(1,30),
    "time_on_site" : np.random.normal(5,2,n).clip(1,15),
    "recency" : np.random.normal(10,2,n).clip(1,100)
})

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

inertia =[]
for k in range(1,11):
    model = KMeans(n_clusters=k,random_state=42)
    model.fit(scaled_data)
    inertia.append(model.inertia_)

plt.plot(range(1,11),inertia,marker="o")
plt.xlabel("Küme Sayısı (k)")
plt.ylabel("Inertia")
plt.title("Elbow Yöntemi")
plt.show()

best_k = find_elbow(inertia)
print("En iyi k değeri:", best_k)

model = KMeans(n_clusters=best_k,random_state=42)
clusters = model.fit_predict(scaled_data)

data["cluster"] = clusters

cluster_summary = data.groupby("cluster").mean().round(2)
print(cluster_summary)


