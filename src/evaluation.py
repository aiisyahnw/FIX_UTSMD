"""
MD - MID EXAM – Step 4: Model Evaluation

Evaluates:
1. Classification model (placement_status)
2. Regression model (salary_lpa)
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import numpy as np


def evaluate(test_data, clf_pipeline, reg_pipeline):

    X_test, y_clf_test, y_reg_test = test_data

    #1. Classification Evaluation
    y_pred_clf = clf_pipeline.predict(X_test)

    acc  = accuracy_score(y_clf_test, y_pred_clf)
    prec = precision_score(y_clf_test, y_pred_clf)
    rec  = recall_score(y_clf_test, y_pred_clf)
    f1   = f1_score(y_clf_test, y_pred_clf)

    print("-" * 30)
    print("Classification Results:")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}  <-- main metric")
    print("-" * 30)

    #2. Regression Evaluation
    #hanya untuk yang placed
    mask = y_clf_test == 1

    X_test_reg = X_test[mask]
    y_test_reg = y_reg_test[mask]

    y_pred_reg = reg_pipeline.predict(X_test_reg)

    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_reg, y_pred_reg)

    print("\n" + "-" * 30)
    print("Regression Results:")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")
    print("-" * 30)

    return (acc, prec, rec, f1), (mae, rmse, r2)


if __name__ == "__main__":
    print("Run evaluation from pipeline")