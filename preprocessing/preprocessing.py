from sklearn.preprocessing import StandardScaler

def process_data(X):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled,scaler

