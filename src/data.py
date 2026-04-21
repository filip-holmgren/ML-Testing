from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd
import numpy as np

from src.clean_counter import clean_counter
from src.preprocess import transform


def load_data(path: str) -> tuple[pd.DataFrame, np.ArrayLike, dict]:
    data_set = pd.read_csv(path)

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
    return X, y, category_maps


def split_data(X: pd.DataFrame, y: np.ArrayLike, test_size: int, random_state: int) -> list:
    return train_test_split(
        X, y, random_state=random_state, stratify=y, test_size=test_size
    )


def apply_smote(X_train: any, y_train: any, random_state: int) -> any:
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Resampled class distribution: {clean_counter(Counter(y_train_res))}")
    return X_train_res, y_train_res
