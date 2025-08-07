import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
X_scaled = scaler.fit_transform()

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

#yapay sinir ağı modelini kur




