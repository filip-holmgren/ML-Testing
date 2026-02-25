from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings

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

    params = {
        "objective": "reg:squarederror", 
        "tree_method": "auto", 
        "learning_rate": 0.01, 
        "max_depth": 12,
        "subsample": 0.6,
        "colsample_bytree": 0.8
    }

    evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

    n = 10000

    results = xgb.cv(
        params, dtrain_reg,
        num_boost_round=n,
        nfold=5,
        early_stopping_rounds=20
    )

    best_rmse = results['test-rmse-mean'].min()
    print(results.head())
    print(best_rmse)

    best_n_rounds = results['test-rmse-mean'].argmin() + 1
    verbose_eval = best_n_rounds // 10

    print(f"Best num boost rounds {best_n_rounds}")
    print(f"Verbose eval every {verbose_eval} rounds")

    model = xgb.train(
        params,
        dtrain_reg,
        num_boost_round = best_n_rounds,
        evals=evals,
        verbose_eval = verbose_eval
    )

    preds = model.predict(dtest_reg)

    rmse = root_mean_squared_error(y_test, preds)
    print(f"RMSE of the base model: {rmse:.3f}")


if __name__ == "__main__":
    main()
