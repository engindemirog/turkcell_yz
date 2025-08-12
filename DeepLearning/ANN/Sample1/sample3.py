import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Reprodüksiyon
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 1) Veriyi yükle ve temizle
df = pd.read_csv("Telco-Customer-Churn.csv")
df.drop(["customerID"], axis=1, inplace=True)

# Sayısallaştırma / temizlik
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Binary kolonlar
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

# Kalan kategorikler için one-hot
df = pd.get_dummies(df, drop_first=False)

# 2) Giriş-çıkış ayır
X = df.drop("Churn", axis=1)
y = df["Churn"].astype(int)

# 3) Train/Test ayır (önce ayır, sonra ölçekle)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

# 4) Ölçekleme (yalnızca train'e fit et)
scaler = StandardScaler(with_mean=False)  # sparse ihtimaline karşı güvenli seçim
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 5) SMOTE (yalnızca eğitim seti)
smote = SMOTE(random_state=SEED, k_neighbors=5)
X_train_bal, y_train_bal = smote.fit_resample(X_train_s, y_train)

# 6) Model
model = Sequential()
model.add(Dense(64, input_dim=X_train_bal.shape[1], activation="relu"))
model.add(Dropout(0.30))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.20))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=1e-3), metrics=["accuracy"])

# 7) Eğitim (artık class_weight kullanmıyoruz; dengeleme SMOTE ile yapıldı)
history = model.fit(
    X_train_bal,
    y_train_bal,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# 8) Değerlendirme
y_pred_proba = model.predict(X_test_s)
y_pred = (y_pred_proba > 0.5).astype(int)

print("Confusion matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))