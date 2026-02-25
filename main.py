from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import optuna

def main():
    warnings.filterwarnings("ignore")
    diamonds = sns.load_dataset("diamonds")

    X, y = diamonds.drop('price', axis=1), diamonds[['price']]

    cats = X.select_dtypes(exclude=np.number).columns.tolist()
    for col in cats:
        X[col] = X[col].astype('category')

    diamonds['volume'] = diamonds['x'] * diamonds['y'] * diamonds['z']


    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

    def objective(trial): 
        params = {
            "objective": "reg:squarederror", 
            "tree_method": "hist",
            "device": "cuda",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        } # 522 latest


        results = xgb.cv(
            params, dtrain_reg,
            num_boost_round=10000,
            nfold=5,
            early_stopping_rounds=20,
            verbose_eval = False
        )

        return results["test-rmse-mean"].min()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Best parameters: ", study.best_params)
    print("Best RMSE (CV): ", study.best_value)

    best_params = {
        "objective": "reg:squarederror", 
        "tree_method": "hist",
        "device": "cuda",
        **study.best_params
    }

    cv_results = xgb.cv(
        best_params,
        dtrain_reg,
        num_boost_round=1000,
        nfold=5,
        early_stopping_rounds=20,
        verbose_eval=False
    )

    evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

    best_rmse = cv_results['test-rmse-mean'].min()
    print(cv_results.head())
    print(best_rmse)

    best_n_rounds = cv_results['test-rmse-mean'].argmin() + 1
    verbose_eval = best_n_rounds // 10

    print(f"Best num boost rounds {best_n_rounds}")
    print(f"Verbose eval every {verbose_eval} rounds")

    model = xgb.train(
        best_params,
        dtrain_reg,
        num_boost_round = best_n_rounds,
        evals=evals,
        verbose_eval = verbose_eval
    )

    preds = model.predict(dtest_reg)

    rmse = root_mean_squared_error(y_test, preds)
    print(f"RMSE of the final model: {rmse:.3f}")


if __name__ == "__main__":
    main()
