"""Stacking ensemble architecture: RF + XGBoost + LightGBM + CatBoost base
learners feeding a Logistic Regression meta-learner, wrapped together with
the shared preprocessing pipeline."""

from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ffsm import config
from ffsm.pipeline import build_preprocessor


def build_base_learners() -> list[tuple[str, object]]:
    return [
        ("random_forest", RandomForestClassifier(**config.RF_PARAMS)),
        ("xgboost", XGBClassifier(**config.XGB_PARAMS)),
        ("lightgbm", LGBMClassifier(**config.LGBM_PARAMS)),
        ("catboost", CatBoostClassifier(**config.CATBOOST_PARAMS)),
    ]


def build_meta_learner() -> LogisticRegression:
    return LogisticRegression(**config.META_LEARNER_PARAMS)


def build_stacking_classifier() -> StackingClassifier:
    """StackingClassifier with cross-validated (out-of-fold) base-learner
    predictions feeding the meta-learner, which avoids meta-learner
    overfitting to the base learners' training-set predictions."""
    return StackingClassifier(
        estimators=build_base_learners(),
        final_estimator=build_meta_learner(),
        cv=config.CV_FOLDS,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )


def build_full_pipeline() -> Pipeline:
    """Preprocessing + stacking ensemble as a single fit/predict object."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("stack", build_stacking_classifier()),
        ]
    )
