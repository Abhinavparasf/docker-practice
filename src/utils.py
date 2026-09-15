import re
import pandas as pd


def process_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.dropna()
    df_clean["clean_email"] = df_clean["email"].apply(process_text)
    return df_clean

