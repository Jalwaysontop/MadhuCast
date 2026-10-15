"""
Baseline Models
===============
Three baseline models for the ablation table, so Person 2's deep fusion
model has proper comparisons.

Baselines (from spec Section 7):
  1. Persistence  — predict the last known glucose value
  2. Linear ARX   — autoregressive with exogenous inputs
  3. LightGBM     — gradient boosting on lag features

All baselines share the same fit / predict / evaluate interface so
Person 2's evaluation harness (src/evaluation/metrics.py) can run them
with identical splits and seeds.

Person: Person 3 (Data Pipeline)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature columns used by ARX and LightGBM baselines
# These must match the keys produced by feature_builder.py
FEATURE_COLUMNS = [
    # Glucose features
    "glucose_mean", "glucose_std", "glucose_min", "glucose_max",
    "glucose_range", "glucose_last", "glucose_roc_5", "glucose_roc_15",
    "glucose_cv", "glucose_trend",
    # HR features
    "hr_mean", "hr_std", "hr_min", "hr_max", "hr_resting_proxy",
    # Activity features
    "steps_total", "steps_mean_per_interval",
    "activity_intensity_proxy", "active_fraction",
    # Sleep features
    "sleep_last_duration_min", "sleep_hours_ago",
    "sleep_efficiency", "slept_last_night",
    # Meal features
    "recent_meal_gl", "recent_meal_carbs_g", "recent_meal_calories",
    "time_since_last_meal_min", "cumulative_carbs_4h", "meal_count_4h",
    # EHR static features
    "age", "sex_male", "bmi", "hba1c", "fasting_glucose",
    "total_cholesterol", "ldl", "hdl", "triglycerides",
    "genetic_risk_score", "diabetes_duration_years",
    "comorbidity_hypertension", "comorbidity_ckd",
    "comorbidity_cvd", "comorbidity_retinopathy",
    "comorbidity_neuropathy",
]

# Target columns
TARGET_COLUMNS = ["glucose_target_30", "glucose_target_60", "glucose_target_120"]
EVENT_COLUMNS  = ["spike_target", "hypo_target"]


# ═════════════════════════════════════════════════════════════════════════════
#  Abstract base class
# ═════════════════════════════════════════════════════════════════════════════

class BaselineModel(ABC):
    """
    Abstract interface that all baselines implement.

    Person 2's evaluation harness calls:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        proba = model.predict_proba(X_test)   # for event heads
    """

    def __init__(self, name: str):
        self.name = name
        self._is_fitted = False

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> "BaselineModel":
        """Fit the model. X has feature columns, y has target columns."""
        ...

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict glucose values at 30/60/120 min.

        Returns DataFrame with columns matching TARGET_COLUMNS.
        """
        ...

    @abstractmethod
    def predict_proba(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict event probabilities (spike, hypo).

        Returns DataFrame with columns matching EVENT_COLUMNS,
        values in [0, 1].
        """
        ...

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convenience: fit on train, predict on test."""
        self.fit(X_train, y_train)
        return self.predict(X_test), self.predict_proba(X_test)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self._is_fitted})"


# ═════════════════════════════════════════════════════════════════════════════
#  1. Persistence Baseline
# ═════════════════════════════════════════════════════════════════════════════

class PersistenceBaseline(BaselineModel):
    """
    Predict the last known glucose value for all future horizons.

    This is the simplest possible baseline — any model worth deploying
    must beat it.
    """

    def __init__(self):
        super().__init__(name="persistence")
        self._spike_threshold = 180.0
        self._hypo_threshold  = 70.0

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "PersistenceBaseline":
        """No fitting needed — persistence just uses the last glucose value."""
        self._is_fitted = True
        logger.info("PersistenceBaseline: fit called (no-op)")
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict: last known glucose for all horizons."""
        last_glucose = X["glucose_last"].values
        preds = pd.DataFrame({
            "glucose_target_30":  last_glucose,
            "glucose_target_60":  last_glucose,
            "glucose_target_120": last_glucose,
        })
        return preds

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Event probability from last glucose value.

        Simple heuristic: probability increases as glucose approaches threshold.
        """
        last = X["glucose_last"].values

        # Spike probability: sigmoid-like based on distance to threshold
        spike_dist = (last - self._spike_threshold) / 30  # scale
        spike_prob = 1 / (1 + np.exp(-spike_dist))

        # Hypo probability
        hypo_dist  = (self._hypo_threshold - last) / 15
        hypo_prob  = 1 / (1 + np.exp(-hypo_dist))

        return pd.DataFrame({
            "spike_target": spike_prob,
            "hypo_target":  hypo_prob,
        })


# ═════════════════════════════════════════════════════════════════════════════
#  2. Linear ARX Baseline
# ═════════════════════════════════════════════════════════════════════════════

class LinearARXBaseline(BaselineModel):
    """
    Auto-Regressive with eXogenous inputs (ARX) using linear regression.

    Uses glucose lags + selected exogenous features to predict future glucose.
    Simpler than deep models but captures linear trends.
    """

    def __init__(self):
        super().__init__(name="linear_arx")
        self._models: Dict[str, Any] = {}
        self._event_models: Dict[str, Any] = {}
        self._feature_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "LinearARXBaseline":
        """Fit separate linear models for each horizon + event."""
        from sklearn.linear_model import Ridge, LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        self._feature_cols = [c for c in FEATURE_COLUMNS if c in X.columns]
        X_clean = X[self._feature_cols].copy()
        X_clean = X_clean.fillna(X_clean.median())

        # Regression models for glucose targets
        for target_col in TARGET_COLUMNS:
            if target_col not in y.columns:
                continue
            y_col = y[target_col].values
            valid_mask = ~np.isnan(y_col)

            if valid_mask.sum() < 10:
                logger.warning("ARX: Too few valid samples for %s", target_col)
                continue

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ])
            pipe.fit(X_clean[valid_mask], y_col[valid_mask])
            self._models[target_col] = pipe

        # Classification models for events
        for event_col in EVENT_COLUMNS:
            if event_col not in y.columns:
                continue
            y_event = y[event_col].astype(int).values
            if len(np.unique(y_event)) < 2:
                logger.warning("ARX: Only one class for %s, skipping", event_col)
                continue

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                )),
            ])
            pipe.fit(X_clean, y_event)
            self._event_models[event_col] = pipe

        self._is_fitted = True
        logger.info(
            "LinearARXBaseline: fitted %d regression + %d classification models",
            len(self._models), len(self._event_models)
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict glucose at each horizon."""
        X_clean = X[self._feature_cols].copy().fillna(X[self._feature_cols].median())
        preds = {}

        for target_col in TARGET_COLUMNS:
            if target_col in self._models:
                preds[target_col] = self._models[target_col].predict(X_clean)
            else:
                # Fallback to persistence
                preds[target_col] = X["glucose_last"].values if "glucose_last" in X.columns else np.full(len(X), np.nan)

        return pd.DataFrame(preds)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict event probabilities."""
        X_clean = X[self._feature_cols].copy().fillna(X[self._feature_cols].median())
        proba = {}

        for event_col in EVENT_COLUMNS:
            if event_col in self._event_models:
                proba[event_col] = self._event_models[event_col].predict_proba(X_clean)[:, 1]
            else:
                proba[event_col] = np.full(len(X), 0.5)

        return pd.DataFrame(proba)


# ═════════════════════════════════════════════════════════════════════════════
#  3. LightGBM Baseline
# ═════════════════════════════════════════════════════════════════════════════

class LightGBMBaseline(BaselineModel):
    """
    Gradient Boosted Decision Trees on lag + exogenous features.

    Typically the strongest classical baseline for time-series regression
    with tabular features.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        seed: int = 42,
    ):
        super().__init__(name="lightgbm")
        self._params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "random_state": seed,
        }
        self._models: Dict[str, Any] = {}
        self._event_models: Dict[str, Any] = {}
        self._feature_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "LightGBMBaseline":
        """Fit LightGBM models for each target."""
        import lightgbm as lgb

        self._feature_cols = [c for c in FEATURE_COLUMNS if c in X.columns]
        X_clean = X[self._feature_cols].copy()
        # LightGBM handles NaN natively — no need to fill

        # Regression models
        for target_col in TARGET_COLUMNS:
            if target_col not in y.columns:
                continue
            y_col = y[target_col].values
            valid_mask = ~np.isnan(y_col)

            if valid_mask.sum() < 10:
                logger.warning("LightGBM: Too few valid samples for %s", target_col)
                continue

            model = lgb.LGBMRegressor(
                **self._params,
                verbose=-1,
            )
            model.fit(
                X_clean[valid_mask], y_col[valid_mask],
                eval_set=None,  # Use validation split separately
            )
            self._models[target_col] = model

        # Classification models for events
        for event_col in EVENT_COLUMNS:
            if event_col not in y.columns:
                continue
            y_event = y[event_col].astype(int).values

            if len(np.unique(y_event)) < 2:
                logger.warning("LightGBM: Only one class for %s, skipping", event_col)
                continue

            # Compute scale_pos_weight for imbalanced classes
            n_pos = y_event.sum()
            n_neg = len(y_event) - n_pos
            scale_weight = n_neg / max(n_pos, 1)

            model = lgb.LGBMClassifier(
                **self._params,
                scale_pos_weight=scale_weight,
                verbose=-1,
            )
            model.fit(X_clean, y_event)
            self._event_models[event_col] = model

        self._is_fitted = True
        logger.info(
            "LightGBMBaseline: fitted %d regression + %d classification models "
            "(n_estimators=%d, lr=%.3f)",
            len(self._models), len(self._event_models),
            self._params["n_estimators"], self._params["learning_rate"],
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict glucose at each horizon."""
        X_clean = X[self._feature_cols].copy()
        preds = {}

        for target_col in TARGET_COLUMNS:
            if target_col in self._models:
                preds[target_col] = self._models[target_col].predict(X_clean)
            else:
                preds[target_col] = X["glucose_last"].values if "glucose_last" in X.columns else np.full(len(X), np.nan)

        return pd.DataFrame(preds)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict event probabilities."""
        X_clean = X[self._feature_cols].copy()
        proba = {}

        for event_col in EVENT_COLUMNS:
            if event_col in self._event_models:
                proba[event_col] = self._event_models[event_col].predict_proba(X_clean)[:, 1]
            else:
                proba[event_col] = np.full(len(X), 0.5)

        return pd.DataFrame(proba)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return feature importance from the 30-min glucose model.

        Useful for sanity-checking which features drive predictions.
        """
        target = "glucose_target_30"
        if target not in self._models:
            return pd.DataFrame()

        model = self._models[target]
        importance = model.feature_importances_
        fi = pd.DataFrame({
            "feature": self._feature_cols,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        return fi


# ═════════════════════════════════════════════════════════════════════════════
#  Factory and utilities
# ═════════════════════════════════════════════════════════════════════════════

def get_all_baselines(seed: int = 42) -> List[BaselineModel]:
    """Return all baseline models for the ablation table."""
    return [
        PersistenceBaseline(),
        LinearARXBaseline(),
        LightGBMBaseline(seed=seed),
    ]


def prepare_Xy(
    windows: List[Dict[str, Any]],
    split_ids: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare X (features) and y (targets) DataFrames from windows,
    filtered to the given split participant IDs.

    Parameters
    ----------
    windows  : List of window dicts from feature_builder.build_windows().
    split_ids: Participant IDs for this split.

    Returns
    -------
    X : DataFrame with FEATURE_COLUMNS
    y : DataFrame with TARGET_COLUMNS + EVENT_COLUMNS
    """
    rows_x = []
    rows_y = []

    for w in windows:
        if w["participant_id"] not in split_ids:
            continue

        row_x = {col: w.get(col, np.nan) for col in FEATURE_COLUMNS}
        row_y = {}
        for col in TARGET_COLUMNS + EVENT_COLUMNS:
            row_y[col] = w.get(col, np.nan)

        rows_x.append(row_x)
        rows_y.append(row_y)

    X = pd.DataFrame(rows_x)
    y = pd.DataFrame(rows_y)

    # Convert event columns to int (with NaN handling)
    for col in EVENT_COLUMNS:
        if col in y.columns:
            y[col] = y[col].fillna(0).astype(int)

    return X, y


def run_all_baselines(
    windows: List[Dict[str, Any]],
    splits: Dict[str, List[str]],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run all baseline models and produce a summary results table.

    Returns a DataFrame with one row per model per metric.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    X_train, y_train = prepare_Xy(windows, splits["train"])
    X_val, y_val     = prepare_Xy(windows, splits["val"])
    X_test, y_test   = prepare_Xy(windows, splits["test"])

    baselines = get_all_baselines(seed)
    results: List[Dict] = []

    for model in baselines:
        logger.info("Running baseline: %s", model.name)

        try:
            model.fit(X_train, y_train)
        except Exception as e:
            logger.error("Failed to fit %s: %s", model.name, e)
            continue

        # Evaluate on test set
        try:
            preds = model.predict(X_test)
        except Exception as e:
            logger.error("Failed to predict with %s: %s", model.name, e)
            continue

        for target_col in TARGET_COLUMNS:
            if target_col not in preds.columns or target_col not in y_test.columns:
                continue
            y_true = y_test[target_col].values
            y_pred = preds[target_col].values

            # Filter valid
            valid = ~(np.isnan(y_true) | np.isnan(y_pred))
            if valid.sum() < 5:
                continue

            mae  = mean_absolute_error(y_true[valid], y_pred[valid])
            rmse = np.sqrt(mean_squared_error(y_true[valid], y_pred[valid]))

            horizon = target_col.replace("glucose_target_", "") + " min"
            results.append({
                "model":   model.name,
                "horizon": horizon,
                "MAE":     round(mae, 2),
                "RMSE":    round(rmse, 2),
                "n_test":  int(valid.sum()),
            })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        logger.info("\n=== Baseline Results ===\n%s", results_df.to_string(index=False))

    return results_df
