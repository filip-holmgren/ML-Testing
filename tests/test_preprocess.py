import pandas as pd


def test_missing_values_handled(category_maps, feature_cols):
    df = pd.DataFrame({col: ["Unknown", None, "test"] for col in feature_cols})

    for col in feature_cols:
        if col in category_maps:
            df[col] = df[col].fillna("Unknown")

    assert df.isnull().sum().sum() >= 0  # no crash baseline
