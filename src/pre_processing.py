"""
MD - MID EXAM – Step 2: Data Preprocessing
Applies cleaning, encoding, and prepares data for modeling.
"""

import os
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline


def preprocess():
    os.makedirs("artifacts", exist_ok=True)

    BASE_DIR = Path(__file__).parent
    DATA_PATH = BASE_DIR / "ingested" / "A_ingested.csv"

    df = pd.read_csv(DATA_PATH)

    #1. Drop column
    df = df.drop(columns=["Student_ID"], errors='ignore')

    #2. Handle missing
    df["extracurricular_involvement"] = df["extracurricular_involvement"].fillna("Missing")

    mapping = {
        "Missing": -1,
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    df["extracurricular_involvement"] = df["extracurricular_involvement"].map(mapping)

    #3. Feature groups
    num_cols = [
        'cgpa', 'tenth_percentage', 'twelfth_percentage',
        'backlogs', 'study_hours_per_day', 'attendance_percentage',
        'projects_completed', 'internships_completed',
        'coding_skill_rating', 'communication_skill_rating',
        'aptitude_skill_rating', 'certifications_count',
        'extracurricular_involvement', 'sleep_hours'
    ]

    ordinal_cols = ['family_income_level', 'city_tier']

    nominal_cols = ['gender', 'branch', 'part_time_job', 'internet_access']

    #4. split X & y
    X = df[num_cols + ordinal_cols + nominal_cols]

    y_clf = df['placement_status'].map({
        'Not Placed': 0,
        'Placed': 1
    })

    y_reg = df['salary_lpa']

    #5. Pipelines  
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    ord_pipeline = Pipeline([
        ('encoder', OrdinalEncoder(categories=[
            ["Low", "Medium", "High"],
            ["Tier 3", "Tier 2", "Tier 1"]
        ]))
    ])

    nom_pipeline = Pipeline([
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    #6. Combine
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('ord', ord_pipeline, ordinal_cols),
        ('nom', nom_pipeline, nominal_cols)
    ])

    #7. Split data
    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
        X, y_clf, y_reg,
        test_size=0.2,
        random_state=42,
        stratify=y_clf
    )

    return (
        (X_train, y_clf_train, y_reg_train),
        (X_test, y_clf_test, y_reg_test),
        preprocessor
    )


if __name__ == "__main__":
    preprocess()