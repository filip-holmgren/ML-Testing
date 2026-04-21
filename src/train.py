from sklearn.metrics import f1_score
import pandas as pd
import xgboost as xgb
import numpy as np
import optuna

from src.config import Config

def train_model(X_train_res: any, y_train_res: any, X_test: any, y_test: any, config: Config) -> tuple[xgb.Booster, any]:
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
            nfold=config.num_folds,
            num_boost_round=config.max_rounds,
            early_stopping_rounds=config.early_stopping_rounds,
            verbose_eval=config.verbose,
        )

        return cv["test-logloss-mean"].min()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=config.num_studies)

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
        nfold=config.num_folds,
        num_boost_round=config.max_rounds,
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=config.verbose,
    )
    best_n_rounds = cv_results.shape[0]
    print(f"Best number of boosting rounds: {best_n_rounds}")

    # Train final model
    model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=best_n_rounds,
        evals=[(dtrain, "train"), (dtest, "validation")],
        verbose_eval=max(1, best_n_rounds // 10),
    )

    y_prob = model.predict(dtest)

    best_thresh = optimize_threshold(y_test, y_prob)

    return model, best_thresh

def optimize_threshold(y_test: any, y_prob: pd.ndarray) -> any:
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        pred = (y_prob > thresh).astype(int)
        f1 = f1_score(y_test, pred)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    
    print(f"Optimal threshold for class 1: {best_thresh:.2f}, F1: {best_f1:.4f}")

    return best_thresh