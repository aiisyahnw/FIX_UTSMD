"""
MD - MID EXAM – Step 3: Model Training

Trains:
1. Classification model (placement_status)
2. Regression model (salary_lpa, only for placed students)
"""

import os
import joblib
import numpy as np

from xgboost import XGBClassifier, XGBRegressor
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from sklearn.model_selection import GridSearchCV

import mlflow
import mlflow.sklearn

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

mlflow.set_experiment("Student Placement")
def train(train_data, preprocessor):

    os.makedirs("artifacts", exist_ok=True)

    X_train, y_clf_train, y_reg_train = train_data

    #2. Classification (XGBoost + best params)
    clf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('resample', SMOTETomek(random_state=42)),
        ('model', XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            n_jobs=-1,
            verbosity=0
        ))
    ])


    with mlflow.start_run(run_name="XGB_Classification"):

        grid = GridSearchCV(
            clf_pipeline,
            param_grid={
                'model__n_estimators': [100, 200],
                'model__max_depth': [3, 5],
               'model__learning_rate': [0.05, 0.1]
            },
            cv=3,
            scoring='f1',
            n_jobs=-1
        )

        grid.fit(X_train, y_clf_train)
        clf_pipeline = grid.best_estimator_

        y_pred_clf = clf_pipeline.predict(X_train)

        # Metrics
        acc = accuracy_score(y_clf_train, y_pred_clf)
        prec = precision_score(y_clf_train, y_pred_clf)
        rec = recall_score(y_clf_train, y_pred_clf)
        f1 = f1_score(y_clf_train, y_pred_clf)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.log_param("model", "XGBClassifier")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("eval_metric", "logloss")
        mlflow.log_param("n_jobs", -1)
        mlflow.log_param("verbosity", 0)

        mlflow.log_param("resampling", "SMOTETomek")

        mlflow.sklearn.log_model(clf_pipeline, "classification_model")

    joblib.dump(clf_pipeline, "artifacts/classification_model.pkl")
    print("Classification model trained & saved.")

    #3. Regression (XGBoost + best params)
    mask = y_clf_train == 1

    X_train_reg = X_train[mask]
    y_train_reg = y_reg_train[mask]

    reg_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(
            random_state=42,
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample = 0.8,
            n_jobs=-1,
            verbosity=0
        ))
    ])
    with mlflow.start_run(run_name="XGB_Regression"):

        reg_pipeline.fit(X_train_reg, y_train_reg)

        y_pred_reg = reg_pipeline.predict(X_train_reg)

        mae = mean_absolute_error(y_train_reg, y_pred_reg)
        mse = mean_squared_error(y_train_reg, y_pred_reg)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_train_reg, y_pred_reg)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.log_param("model", "XGBoost_Regressor")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("subsample", 0.8)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_jobs", -1)
        mlflow.log_param("verbosity", 0)

        mlflow.sklearn.log_model(reg_pipeline, "regression_model")

    joblib.dump(reg_pipeline, "artifacts/regression_model.pkl")
    print("Regression model trained & saved.")

    return clf_pipeline, reg_pipeline