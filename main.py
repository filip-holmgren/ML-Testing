from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import optuna

def main():
    warnings.filterwarnings("ignore")
    data_set = pd.read_csv("training_data.csv")

    X, y = data_set.drop('label', axis=1), data_set[['label']]

    y_encoded = OrdinalEncoder().fit_transform(y)

    cats = X.select_dtypes(exclude=np.number).columns.tolist()
    for col in cats:
        X[col] = X[col].astype('category')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state=1, stratify=y_encoded)

    dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, y_test, enable_categorical=True)

    scale_pos_weight = len(y_train[y_train == 1]) / len(y_train[y_train == 0])
    print(scale_pos_weight)

    def objective(trial): 
        params = {
            "objective": "multi:softprob",
            "num_class": 5,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        cv_results = xgb.cv(
            params, 
            dtrain,
            num_boost_round=2000,
            nfold=5,
            early_stopping_rounds=20,
            verbose_eval = False
        )

        return cv_results["test-mlogloss-mean"].max()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    print("Best parameters: ", study.best_params)
    print("Best RMSE (CV): ", study.best_value)

    best_params = {
        "objective": "multi:softprob", 
        "tree_method": "hist",
        "num_class": 5,
        "scale_pos_weight": scale_pos_weight,
        **study.best_params
    }

    cv_results = xgb.cv(
        best_params,
        dtrain,
        num_boost_round=200,
        nfold=5,
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    best_n_rounds = cv_results["test-mlogloss-mean"].argmin() + 1
    print(f"Best number of boosting rounds: {best_n_rounds}")

    model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=best_n_rounds,
        evals=[(dtrain, "train"), (dtest, "validation")],
        verbose_eval=best_n_rounds // 10,
    )
    
    preds_proba = model.predict(dtest)
    preds = np.argmax(preds_proba, axis=1)

    accuracy = accuracy_score(y_test, preds)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    main()
