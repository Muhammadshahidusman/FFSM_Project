"""Shared preprocessing: scaling of continuous factors + one-hot encoding of
the categorical (class-code) factors, used identically for every base
learner inside the stack and at raster-prediction time."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ffsm import config


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: StandardScaler on continuous factors,
    OneHotEncoder on categorical (lulc, soiltext) factors."""
    return ColumnTransformer(
        transformers=[
            (
                "continuous",
                StandardScaler(),
                config.CONTINUOUS_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Feature names after the ColumnTransformer has been fitted — needed to
    label SHAP plots meaningfully (one-hot columns get expanded names)."""
    return list(preprocessor.get_feature_names_out())
