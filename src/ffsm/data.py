"""Loading and splitting the point-sampled training data."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from ffsm import config


def load_training_table(csv_path=None) -> pd.DataFrame:
    """Load the clean flood / non-flood training points CSV."""
    csv_path = csv_path or config.TRAINING_CSV
    df = pd.read_csv(csv_path)

    missing = [c for c in config.ALL_FEATURES + [config.TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Training CSV at {csv_path} is missing expected column(s): {missing}"
        )

    # Cast categorical class-code columns to plain (numeric) integers. Kept
    # numeric rather than string so shap's tabular masker (np.isfinite /
    # np.isclose over the whole row) works without special-casing — the
    # OneHotEncoder in the pipeline treats each integer code as its own
    # category either way.
    for c in config.CATEGORICAL_FEATURES:
        df[c] = df[c].round().astype(int)

    return df


def get_feature_target_split(df: pd.DataFrame):
    X = df[config.ALL_FEATURES].copy()
    y = df[config.TARGET_COL].astype(int).copy()
    return X, y


def train_test_split_data(X, y, test_size=None, random_state=None):
    test_size = test_size if test_size is not None else config.TEST_SIZE
    random_state = random_state if random_state is not None else config.RANDOM_STATE
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
