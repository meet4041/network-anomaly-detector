from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np

def get_isolation_forest_model():
    return IsolationForest(n_estimators=100, contamination='auto', random_state=42)

def get_autoencoder_model(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(20, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def predict_autoencoder(model, X_scaled):
    model.fit(X_scaled, X_scaled, epochs=5, batch_size=256, verbose=0)
    reconstructions = model.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, 95)
    preds = [1 if e > threshold else 0 for e in mse]
    return preds, mse, threshold
