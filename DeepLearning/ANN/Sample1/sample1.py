import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

from sklearn.metrics import classification_report,confusion_matrix


#Dataya ulaş
df = pd.read_csv("Telco-Customer-Churn.csv")

df.drop(["customerID"],axis=1,inplace=True)

#Datayı Temizle (Kategorik)(Sayısallaştır)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
df.dropna(inplace=True)


#Binary data temizle(Sayısallaştır)

binary_cols = ["Partner","Dependents","PhoneService","PaperlessBilling","Churn"]

for col in binary_cols:
    df[col] = df[col].map({"Yes":1,"No":0})

df = pd.get_dummies(df)

#Giriş ve çıkışları ayarla

X = df.drop("Churn",axis=1)
y = df["Churn"]


#Sayısal Verileri ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

#yapay sinir ağı modelini kur

model = Sequential() #Sıralı, katman katman öğrenme gerçekleştir

model.add(Dense(64,input_dim=X_train.shape[1],activation="relu")) # Giriş katmanı
model.add(Dropout(0.3)) # nörünların yüzde otuzunu deaktive et, rastgele
model.add(Dense(32,activation="relu")) # gizli katman oluştur, nörün sayısı azaldı
model.add(Dropout(0.2))
model.add(Dense(1,activation="sigmoid")) # çıkış(karar katmanı)

model.compile(loss="binary_crossentropy",optimizer ="adam", metrics=["accuracy"])

history = model.fit(X_train,y_train,validation_split=0.2,epochs = 50, batch_size=32,verbose=1)

y_pred = (model.predict(X_test)>0.5).astype(int)

print("Confusion matrix")
print(confusion_matrix(y_test,y_pred))


print("Classification Report")
print(classification_report(y_test,y_pred))