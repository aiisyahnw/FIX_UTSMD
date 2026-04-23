"""
MD - MID EXAM – Main Pipeline

Runs the full machine learning workflow:
1. Data Ingestion
2. Data Preprocessing
3. Model Training
4. Model Evaluation
"""

from src.data_ingestion import load_data, save_data
from src.pre_processing import preprocess
from src.train import train
from src.evaluation import evaluate


def run_pipeline():

    print("Starting ML Pipeline...")
    print("=" * 40)

    #1. Data Ingestion
    print("Step 1: Data Ingestion...")
    df = load_data()
    save_data(df)

    #2. Preprocessing
    print("Step 2: Data Preprocessing...")
    train_data, test_data, preprocessor = preprocess()

    #3. Training
    print("Step 3: Model Training...")
    clf_model, reg_model = train(train_data, preprocessor)

    #4. Evaluation
    print("Step 4: Model Evaluation...")
    clf_metrics, reg_metrics = evaluate(test_data, clf_model, reg_model)

    print("=" * 40)
    print("Pipeline completed!")

    #Final Summary
    acc, prec, rec, f1 = clf_metrics
    mae, rmse, r2 = reg_metrics

    print("\nFinal results:")
    print("-" * 30)
    print(f"F1 Score (Classification): {f1:.4f}")
    print(f"RMSE (Regression): {rmse:.4f}")
    print(f"R2 Score (Regression): {r2:.4f}")


if __name__ == "__main__":
    run_pipeline()