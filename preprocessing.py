import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from config import CATEGORICAL_COLUMNS

def preprocess_data(df):
    if 'label' in df.columns:
        df['label'] = df['label'].apply(lambda x: 0 if x == 'normal.' or x == 0 else 1)
    else:
        df['label'] = -1

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop('label', axis=1)
    y = df['label']
    X_scaled = MinMaxScaler().fit_transform(X)
    return X_scaled, y
