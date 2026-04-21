from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
from pathlib import Path
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
import optuna
import json
import shap

from src.preprocess import transform
from src.confusion_matrix_generator import generate_confusion_matrix_visualization

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="model",
        help="Directory where trained model artifacts and metadata will be saved"
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        default="data/training_data.csv",
        help="Path to the input CSV file containing training data"
    )

    parser.add_argument(
        "-s", "--shap",
        action="store_true",
        help="Generate and save SHAP feature importance visualization"
    )

    parser.add_argument(
        "-c", "--confusion-matrix",
        action="store_true",
        help="Generate and save confusion matrix visualization"
    )

    parser.add_argument(
        "--studies",
        type=int,
        default=1000,
        help="Number of Optuna trials for hyperparameter tuning"
    )

    parser.add_argument(
        "--shap-output",
        type=str,
        default="data/shap_visualized_data.png",
        help="File path to save the SHAP visualization image"
    )

    parser.add_argument(
        "--confusion-matrix-output",
        type=str,
        default="data/confusion_matrix.png",
        help="File path to save the confusion matrix image"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Proportion of dataset to use as test set (e.g., 0.25 = 25%%)"
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=1,
        help="Random seed for reproducibility (affects data split and SMOTE)"
    )

    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE oversampling on the training data"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="model.ubj",
        help="Filename for saving the trained model"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging during training and evaluation"
    )

    parser.add_argument(
        "--show-warnings",
        action="store_false",
        help="Disable suppression of warnings (show all warnings if set)"
    )

    parser.add_argument(
        "--max-rounds",
        type=int,
        default=1000,
        help="Maximum number of boosting rounds for XGBoost training"
    )

    parser.add_argument(
        "--early-stopping",
        type=int,
        default=20,
        help="Number of rounds with no improvement before early stopping"
    )

    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation (e.g., 5 = 5-fold CV)"
    )
    
    args = parser.parse_args()

    output_path = args.output
    input_path = args.input
    gen_shap = args.shap
    gen_confusion_matrix = args.confusion_matrix
    num_studies = args.studies
    shap_output_path = args.shap_output
    confusion_matrix_output_path = args.confusion_matrix_output
    test_size = args.test_size
    random_state = args.random_state
    disable_smote = args.no_smote
    model_name = args.model_name
    verbose = args.verbose
    show_warnings = args.show_warnings
    max_rounds = args.max_rounds
    early_stopping_rounds = args.early_stopping
    num_folds = args.num_folds

    if show_warnings:
        warnings.filterwarnings("ignore")
    
    data_set = pd.read_csv(input_path)

    X = data_set.drop("label", axis=1)
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

    print(f"Original class distribution: {clean_counter(Counter(y))}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state, stratify=y, test_size=test_size
    )

    # Apply SMOTE to training data
    if not disable_smote:
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"Resampled class distribution: {clean_counter(Counter(y_train_res))}")
    else:
        X_train_res, y_train_res = X_train, y_train

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
            nfold=num_folds,
            num_boost_round=max_rounds,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose,
        )

        return cv["test-logloss-mean"].min()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=num_studies)

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
        nfold=num_folds,
        num_boost_round=max_rounds,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose,
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

    # Predict and optimize threshold
    y_pred_proba = model.predict(dtest)

    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        preds_tmp = (y_pred_proba > thresh).astype(int)
        f1 = f1_score(y_test, preds_tmp)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    y_pred = (y_pred_proba > best_thresh).astype(int)
    print(f"Optimal threshold for class 1: {best_thresh:.2f}, F1: {best_f1:.4f}")

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    output_dir = Path(output_path)
    if output_dir.exists() is False:
        output_dir.mkdir(parents=True, exist_ok=True)

    meta = {"feature_cols": list(X.columns)}
    model.save_model(output_dir / model_name)

    with open(output_dir / "config.json", "w") as f:
        f.write(model.save_config())
    with open(output_dir / "category_maps.json", "w") as f:
        json.dump(category_maps, f)
    with open(output_dir / "threshold.json", "w") as f:
        json.dump({"threshold": best_thresh}, f)
    with open(output_dir / "feature_meta.json", "w") as f:
        json.dump(meta, f)

    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    if gen_confusion_matrix:
        Path(confusion_matrix_output_path).parent.mkdir(parents=True, exist_ok=True)
        generate_confusion_matrix_visualization(cm, confusion_matrix_output_path)

    if gen_shap:
        pred = model.predict(dtest, output_margin=True)
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X_test)

        shap_values = explanation.values
        np.abs(shap_values.sum(axis=1) + explanation.base_values - pred).max()

        shap.plots.beeswarm(explanation, max_display=len(X_test.columns), show=False)
        Path(shap_output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(shap_output_path, bbox_inches="tight")

def clean_counter(counter):
    return {int(k): int(v) for k, v in counter.items()}

if __name__ == "__main__":
    main()
