from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_model(X, y):
    models = [
         #("Logistic Regression", LogisticRegression()),
         ("Random Forest", RandomForestClassifier(random_state=42))
    ]

    results = []

    for name, model in models:
        model.fit(X,y)
        return model
    