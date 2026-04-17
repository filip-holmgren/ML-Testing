import json
import pandas as pd


def load_category_maps(path="model/category_maps.json"):
    with open(path) as f:
        return json.load(f)


def transform(df: pd.DataFrame, feature_cols, category_maps):
    df = df.copy()

    for col in feature_cols:
        if col in category_maps:
            df[col] = df[col].fillna("Unknown")
            df[col] = df[col].map(category_maps[col]).fillna(-1).astype(int)

    return df[feature_cols]
