import xgboost as xgb
import pandas as pd
import numpy as np
from src.preprocess import transform


def test_model_loads(model):
    assert model is not None


def test_prediction_shape(model, feature_cols, sample_df, category_maps):
    X = transform(sample_df, feature_cols, category_maps)

    prediction = model.predict(xgb.DMatrix(X))

    assert prediction.shape[0] == len(sample_df)


def test_probability_range(model, feature_cols, sample_df, category_maps):
    X = transform(sample_df, feature_cols, category_maps)

    prediction = model.predict(xgb.DMatrix(X))

    assert prediction.min() >= 0
    assert prediction.max() <= 1


def test_threshold_logic(model, feature_cols, sample_df, category_maps, threshold):
    X = transform(sample_df, feature_cols, category_maps)

    probability = model.predict(xgb.DMatrix(X))
    labels = (probability > threshold).astype(int)

    assert set(labels).issubset({0, 1})


def test_model_on_real_data(model, feature_cols, category_maps):
    df = pd.read_csv("data/training_data.csv")

    df = df.sample(n=50, random_state=42)

    X = transform(df, feature_cols, category_maps)

    prediction = model.predict(xgb.DMatrix(X))

    assert len(prediction) == len(df)
    assert prediction.min() >= 0
    assert prediction.max() <= 1


def test_model_outputs_reasonable_rate(model, feature_cols, category_maps):
    df = pd.read_csv("data/training_data.csv").sample(200, random_state=42)

    X = transform(df, feature_cols, category_maps)
    prediction = model.predict(xgb.DMatrix(X))

    positive_rate = prediction.mean()

    # sanity constraint (adjust based on your domain)
    assert 0.01 < positive_rate < 0.99
