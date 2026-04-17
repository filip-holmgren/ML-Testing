import pytest
import xgboost as xgb
import json
import pandas as pd

MODEL_PATH = "model/model.ubj"
META_PATH = "model/feature_meta.json"
THRESH_PATH = "model/threshold.json"
MAP_PATH = "model/category_maps.json"


@pytest.fixture(scope="session")
def model():
    m = xgb.Booster()
    m.load_model(MODEL_PATH)
    return m


@pytest.fixture(scope="session")
def feature_cols():
    with open(META_PATH) as f:
        return json.load(f)["feature_cols"]


@pytest.fixture(scope="session")
def threshold():
    with open(THRESH_PATH) as f:
        return json.load(f)["threshold"]


@pytest.fixture(scope="session")
def category_maps():
    with open(MAP_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def sample_df():
    return pd.read_csv("data/training_data.csv").head(20)
