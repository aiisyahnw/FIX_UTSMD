"""
MD - MID EXAM – Step 1: Data Ingestion

This module is responsible for loading the raw dataset files (features and targets),
merging them into a single DataFrame, and saving the processed dataset into the
'ingested/' directory for further processing in the pipeline.

The ingestion process ensures that the data is structured and ready for preprocessing
and model training stages.
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
INGESTED_DIR = BASE_DIR / "ingested"
FEATURES_FILE = BASE_DIR.parent / "Data/A.csv"
TARGET_FILE = BASE_DIR.parent / "Data/A_targets.csv"
OUTPUT_FILE = INGESTED_DIR / "A_ingested.csv"

def load_data():
    features = pd.read_csv(FEATURES_FILE)
    targets = pd.read_csv(TARGET_FILE)

    df = pd.concat([features, targets], axis=1)
    
    return df

def save_data(df):
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    df = load_data()
    save_data(df)