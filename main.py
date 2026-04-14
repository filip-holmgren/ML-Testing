from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import optuna

def main():
    warnings.filterwarnings("ignore")
    data_set = pd.read_csv("training_data.csv")

    X, y = data_set.drop('label', axis=1), data_set[['label']]

    # Encode labels to 0/1
    y_encoded = OrdinalEncoder().fit_transform(y).ravel()

    # Encode categorical features
    cats = X.select_dtypes(exclude=np.number).columns.tolist()
    for col in cats:
        X[col] = X[col].astype('category')

    # Fill numeric columns
    num_cols = X.select_dtypes(include=np.number).columns
    X[num_cols] = X[num_cols].fillna(0)

    # Fill categorical columns
    cat_cols = X.select_dtypes(exclude=np.number).columns
    for col in cat_cols:
        X[col] = X[col].cat.add_categories('missing').fillna('missing')
        X[col] = X[col].cat.codes  # string → integer, NaN → -1

    print(f"Original class distribution: {Counter(y_encoded)}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, random_state=1, stratify=y_encoded
    )

    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Resampled class distribution: {Counter(y_train_res)}")

    # Compute scale_pos_weight (optional if using SMOTE)
    scale_pos_weight = len(y_train_res[y_train_res == 0]) / max(len(y_train_res[y_train_res == 1]), 1)
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    dtrain = xgb.DMatrix(X_train_res, y_train_res)
    dtest = xgb.DMatrix(X_test, y_test)

    # Hyperparameter tuning with Optuna
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "eval_metric": "logloss",
            "scale_pos_weight": scale_pos_weight,
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
    study.optimize(objective, n_trials=1000)  # adjust trials as needed

    print("Best hyperparameters:", study.best_params)

    best_params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight,
        **study.best_params
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

    with open("config.json", "w") as f:
        f.write(model.save_config())
    
    model.save_model("model.ubj")

if __name__ == "__main__":
    main()