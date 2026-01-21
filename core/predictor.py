import joblib
import pandas as pd

def load_model(path="models/trained_model.pkl"):
    """
    Load saved model from disk
    """
    model = joblib.load(path)
    return model

def load_encoders(path="models/encoders.pkl"):
    """
    Load saved encoders (optional, if you encoded categorical features)
    """
    try:
        encoders = joblib.load(path)
    except FileNotFoundError:
        encoders = {}
    return encoders

def preprocess_input(df, encoders):
    """
    Encode categorical columns using saved encoders
    """
    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    return df

def predict(model, df, encoders={}):
    """
    df: DataFrame with same features as training
    encoders: dictionary of LabelEncoders used in training
    """
    df_processed = preprocess_input(df, encoders)
    predictions = model.predict(df_processed)
    return predictions
