from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics import confusion_matrix
from pathlib import Path
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import shap

from src.confusion_matrix_generator import generate_confusion_matrix_visualization
from src.config import Config


def save_artifacts(
    model: xgb.Booster,
    best_thresh: any,
    X: pd.DataFrame,
    category_maps: dict,
    output_path: str,
    model_name: str,
) -> None:
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


def evaluate_and_visualize(
    model: xgb.Booster, X_test: any, y_test: any, threshold: any, config: Config
):
    dtest = xgb.DMatrix(X_test)

    y_prob = model.predict(dtest)
    y_pred = (y_prob > threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    if config.gen_confusion_matrix:
        Path(config.confusion_matrix_output_path).parent.mkdir(
            parents=True, exist_ok=True
        )
        generate_confusion_matrix_visualization(cm, config.confusion_matrix_output_path)

    if config.gen_shap:
        pred = model.predict(dtest, output_margin=True)
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X_test)

        shap_values = explanation.values
        np.abs(shap_values.sum(axis=1) + explanation.base_values - pred).max()

        shap.plots.beeswarm(explanation, max_display=len(X_test.columns), show=False)
        Path(config.shap_output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(config.shap_output_path, bbox_inches="tight")
