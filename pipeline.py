"""Leakage-aware training and inference pipeline for GridPulse."""
from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, balanced_accuracy_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "electricity_demand.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics.json"
ERRORS_PATH = ROOT / "outputs" / "misclassified_examples.csv"

# Seven meaningful predictors. The user enters simple values in the app;
# weekend status and previous-hour demand are derived automatically.
FEATURES = [
    "hour", "day_of_week", "is_weekend", "temperature_c", "humidity_pct",
    "current_demand_mw", "demand_lag_1h_mw",
]
TARGET = "spike_next_hour"
REQUIRED_COLUMNS = ["timestamp", *FEATURES, "rolling_mean_24h_mw", "rolling_std_24h_mw", TARGET]


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError("Dataset is missing columns: " + ", ".join(missing))
    df = df[REQUIRED_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Dataset contains invalid timestamps.")
    for col in FEATURES + ["rolling_mean_24h_mw", "rolling_std_24h_mw", TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    if len(df) <= 5000:
        raise ValueError(f"At least 5,001 usable records are required; found {len(df):,}.")
    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found. Clean the dataset before training.")
    if not set(df[TARGET].unique()).issubset({0, 1}) or df[TARGET].nunique() != 2:
        raise ValueError("Target must contain both classes: 0 (normal) and 1 (spike).")
    if not df[FEATURES].apply(np.isfinite).all().all():
        raise ValueError("Predictor columns contain non-finite values.")
    return df


def score_at_threshold(y_true, probability, threshold: float) -> dict:
    pred = (np.asarray(probability) >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def choose_threshold(y_true, probability) -> tuple[float, dict]:
    best_threshold, best_metrics = 0.5, None
    for threshold in np.linspace(0.10, 0.90, 161):
        metrics = score_at_threshold(y_true, probability, float(threshold))
        # Prefer higher precision when F1 ties, to reduce unnecessary alerts.
        rank = (metrics["f1"], metrics["precision"], float(threshold))
        if best_metrics is None or rank > best_metrics["rank"]:
            best_threshold = float(threshold)
            best_metrics = {**metrics, "rank": rank}
    best_metrics.pop("rank", None)
    return best_threshold, best_metrics


def train_models(df: pd.DataFrame | None = None):
    df = load_dataset() if df is None else df.copy()
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)
    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    y_train = train[TARGET].astype(int)
    y_val = validation[TARGET].astype(int)
    y_test = test[TARGET].astype(int)
    if any(part.nunique() < 2 for part in (y_train, y_val, y_test)):
        raise ValueError("Train, validation and test periods must each contain both classes.")

    candidates = {
        "Dummy Baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=350, min_samples_leaf=3, max_features="sqrt",
                class_weight="balanced_subsample", random_state=42, n_jobs=-1,
            )),
        ]),
    }

    fitted = {}
    validation_scores = {}
    test_scores = {}
    thresholds = {}
    for name, model in candidates.items():
        model.fit(train[FEATURES], y_train)
        fitted[name] = model
        if name == "Dummy Baseline":
            val_pred = model.predict(validation[FEATURES])
            val_metrics = score_at_threshold(y_val, val_pred, 0.5)
            threshold = 0.5
            test_pred = model.predict(test[FEATURES])
            test_metrics = score_at_threshold(y_test, test_pred, 0.5)
        else:
            val_probability = model.predict_proba(validation[FEATURES])[:, 1]
            threshold, val_metrics = choose_threshold(y_val, val_probability)
            test_probability = model.predict_proba(test[FEATURES])[:, 1]
            test_metrics = score_at_threshold(y_test, test_probability, threshold)
        thresholds[name] = threshold
        validation_scores[name] = val_metrics
        test_scores[name] = test_metrics

    selected_name = max(
        ["Logistic Regression", "Random Forest"],
        key=lambda name: (validation_scores[name]["f1"], validation_scores[name]["precision"]),
    )
    selected_model = fitted[selected_name]
    selected_threshold = thresholds[selected_name]
    test_probability = selected_model.predict_proba(test[FEATURES])[:, 1]
    test_prediction = (test_probability >= selected_threshold).astype(int)

    test_view = test.copy()
    test_view["actual"] = y_test.to_numpy()
    test_view["predicted"] = test_prediction
    test_view["probability"] = test_probability
    temp_median = float(train["temperature_c"].median())
    test_view["temperature_group"] = np.where(
        test_view["temperature_c"] >= temp_median, "Higher temperature", "Lower temperature"
    )
    demand_median = float(train["current_demand_mw"].median())
    test_view["demand_group"] = np.where(
        test_view["current_demand_mw"] >= demand_median, "Higher current demand", "Lower current demand"
    )
    subgroup = {}
    for column in ("temperature_group", "demand_group"):
        subgroup[column] = {}
        for group, part in test_view.groupby(column, observed=True):
            subgroup[column][str(group)] = {
                "records": int(len(part)),
                "accuracy": round(float(accuracy_score(part.actual, part.predicted)), 4),
                "precision": round(float(precision_score(part.actual, part.predicted, zero_division=0)), 4),
                "recall": round(float(recall_score(part.actual, part.predicted, zero_division=0)), 4),
                "f1": round(float(f1_score(part.actual, part.predicted, zero_division=0)), 4),
                "false_positives": int(((part.actual == 0) & (part.predicted == 1)).sum()),
                "false_negatives": int(((part.actual == 1) & (part.predicted == 0)).sum()),
            }

    artifact = {
        "model": selected_model,
        "model_name": selected_name,
        "threshold": selected_threshold,
        "features": FEATURES,
        "target": TARGET,
        "train_end": str(train.timestamp.max()),
        "validation_end": str(validation.timestamp.max()),
        "test_start": str(test.timestamp.min()),
    }
    report = {
        "dataset_type": "Synthetic educational demonstration data",
        "target_definition": "1 when next-hour demand exceeds 110% of the previous 24-hour mean; otherwise 0",
        "total_rows": int(n),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "class_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "train_end": artifact["train_end"],
        "validation_end": artifact["validation_end"],
        "test_start": artifact["test_start"],
        "validation_results": validation_scores,
        "test_results": test_scores,
        "selected_model": selected_name,
        "selected_threshold": round(float(selected_threshold), 3),
        "selected_metrics": test_scores[selected_name],
        "subgroup_diagnostics": subgroup,
        "classification_report": classification_report(
            y_test, test_prediction, zero_division=0, output_dict=True
        ),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ERRORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    errors = test_view.loc[test_view.actual != test_view.predicted].copy()
    errors.to_csv(ERRORS_PATH, index=False)
    return artifact, report, test_view


def load_or_train():
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        try:
            artifact = joblib.load(MODEL_PATH)
            report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            if report.get("total_rows") == len(load_dataset()):
                return artifact, report
        except Exception:
            pass
    artifact, report, _ = train_models()
    return artifact, report


def build_prediction_row(values: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Build a new row from simple UI inputs plus derived historical context."""
    required = ("hour", "day_of_week", "temperature_c", "humidity_pct", "current_demand_mw")
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError("Missing input(s): " + ", ".join(missing))
    try:
        hour = int(values["hour"])
        temp = float(values["temperature_c"])
        humidity = float(values["humidity_pct"])
        demand = float(values["current_demand_mw"])
    except (TypeError, ValueError):
        raise ValueError("Enter valid numeric values for hour, temperature, humidity and demand.")
    if not 0 <= hour <= 23:
        raise ValueError("Hour must be between 0 and 23.")
    if not -10 <= temp <= 55:
        raise ValueError("Temperature must be between -10°C and 55°C.")
    if not 0 <= humidity <= 100:
        raise ValueError("Humidity must be between 0% and 100%.")
    if not 50 <= demand <= 250:
        raise ValueError("Current demand must be between 50 MW and 250 MW for this demo dataset.")
    day_of_week = int(values["day_of_week"])
    if not 0 <= day_of_week <= 6:
        raise ValueError("Day of week must be between 0 (Monday) and 6 (Sunday).")
    is_weekend = int(day_of_week >= 5)
    previous_demand = float(data.iloc[-1]["current_demand_mw"])
    row = {
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "temperature_c": temp,
        "humidity_pct": humidity,
        "current_demand_mw": demand,
        "demand_lag_1h_mw": previous_demand,
    }
    return pd.DataFrame([row], columns=FEATURES)
