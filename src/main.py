from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
from pathlib import Path
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import optuna
import json

from src.preprocess import transform


def main():
    warnings.filterwarnings("ignore")
    data_set = pd.read_csv("data/training_data.csv")

    X, y = data_set.drop("label", axis=1), data_set[["label"]]
    y = LabelEncoder().fit_transform(data_set["label"])

    categorical_columns = X.select_dtypes(exclude=np.number).columns
    category_maps = {}

    for column in categorical_columns:
        X[column] = X[column].fillna("Unknown")
        X[column] = X[column].astype("category")

        category_maps[column] = {
            category: i for i, category in enumerate(X[column].cat.categories)
        }

    X = transform(X, X.columns, category_maps)

    print(f"Original class distribution: {Counter(y)}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1, stratify=y
    )

    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Resampled class distribution: {Counter(y_train_res)}")

    dtrain = xgb.DMatrix(X_train_res, y_train_res)
    dtest = xgb.DMatrix(X_test, y_test)

    # Hyperparameter tuning with Optuna
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "eval_metric": "logloss",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        cv = xgb.cv(
            params,
            dtrain,
            nfold=5,
            num_boost_round=1000,
            early_stopping_rounds=20,
            verbose_eval=False,
        )

        return cv["test-logloss-mean"].min()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)  # adjust trials as needed

    print("Best hyperparameters:", study.best_params)

    best_params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        **study.best_params,
    }

    # Determine best number of boosting rounds
    cv_results = xgb.cv(
        best_params,
        dtrain,
        nfold=5,
        num_boost_round=1000,
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    best_n_rounds = cv_results["test-logloss-mean"].idxmin() + 1
    print(f"Best number of boosting rounds: {best_n_rounds}")

    # Train final model
    model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=best_n_rounds,
        evals=[(dtrain, "train"), (dtest, "validation")],
        verbose_eval=max(1, best_n_rounds // 10),
    )

    # Predict and optimize threshold
    preds_proba = model.predict(dtest)

    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        preds_tmp = (preds_proba > thresh).astype(int)
        f1 = f1_score(y_test, preds_tmp)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    preds = (preds_proba > best_thresh).astype(int)
    print(f"Optimal threshold for class 1: {best_thresh:.2f}, F1: {best_f1:.4f}")

    # Evaluate
    accuracy = accuracy_score(y_test, preds)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, preds))

    path = Path("model/")
    if path.exists() is False:
        path.mkdir()

    with open("model/config.json", "w") as f:
        f.write(model.save_config())

    model.save_model("model/model.ubj")

    with open("model/category_maps.json", "w") as f:
        json.dump(category_maps, f)

    with open("model/threshold.json", "w") as f:
        json.dump({"threshold": best_thresh}, f)

    meta = {"feature_cols": list(X.columns)}

    with open("model/feature_meta.json", "w") as f:
        json.dump(meta, f)


if __name__ == "__main__":
    main()
