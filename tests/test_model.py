import pytest
import xgboost as xgb
import json
import numpy as np
import pandas as pd
from src.preprocess import transform, load_category_maps


@pytest.fixture(scope="session")
def model():
    model = xgb.Booster()
    model.load_model("model/model.ubj")
    return model


@pytest.fixture(scope="session")
def threshold():
    with open("model/threshold.json") as f:
        return json.load(f)["threshold"]


@pytest.fixture(scope="session")
def feature_cols():
    with open("model/feature_meta.json") as f:
        meta = json.load(f)
    return meta["feature_cols"]


def test_model_loads(model):
    assert model is not None


def test_prediction_shape(model, feature_cols):
    X = pd.DataFrame(np.random.rand(10, len(feature_cols)), columns=feature_cols)
    prediction = model.predict(xgb.DMatrix(X))
    assert prediction.shape[0] == 10


def test_threshold_logic(model, threshold, feature_cols):
    X = pd.DataFrame(np.random.rand(5, len(feature_cols)), columns=feature_cols)
    probability = model.predict(xgb.DMatrix(X))

    labels = (probability > threshold).astype(int)
    assert set(labels).issubset({0, 1})


def test_deterministic(model, feature_cols):
    X = pd.DataFrame(np.random.rand(5, len(feature_cols)), columns=feature_cols)

    p1 = model.predict(xgb.DMatrix(X))
    p2 = model.predict(xgb.DMatrix(X))

    assert np.allclose(p1, p2)


def test_probability_range(model, feature_cols):
    X = pd.DataFrame(np.random.rand(5, len(feature_cols)), columns=feature_cols)
    preds = model.predict(xgb.DMatrix(X))

    assert preds.min() >= 0
    assert preds.max() <= 1


def test_on_real_sample(model, feature_cols):
    df = pd.read_csv("data/training_data.csv").head(20)

    category_maps = load_category_maps()
    X = transform(df[feature_cols], feature_cols, category_maps)

    preds = model.predict(xgb.DMatrix(X))

    assert len(preds) == len(df)
