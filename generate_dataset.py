"""Generate the reproducible synthetic hourly electricity-demand demo dataset.

This dataset is synthetic and is intended for education/demo only. It is not
measured utility data and must not be used for real grid-operation decisions.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "electricity_demand.csv"
SEED = 42


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    # Include one warm-up hour before 2025 and one future hour after 2025 so
    # the first lag and final next-hour target are calculated without leakage.
    timestamps = pd.date_range("2024-12-31 23:00:00", "2026-01-01 00:00:00", freq="h")
    doy = timestamps.dayofyear.to_numpy()
    hour = timestamps.hour.to_numpy()
    dow = timestamps.dayofweek.to_numpy()
    weekend = (dow >= 5).astype(int)

    seasonal_temp = 29.5 + 3.0 * np.sin(2 * np.pi * (doy - 30) / 365.25)
    daily_temp = 3.2 * np.sin(2 * np.pi * (hour - 8) / 24)
    temperature = seasonal_temp + daily_temp + rng.normal(0, 1.1, len(timestamps))
    humidity = np.clip(
        69 - 0.48 * (temperature - 30) + rng.normal(0, 5.0, len(timestamps)),
        42, 95,
    )

    evening_peak = 4.0 * np.exp(-0.5 * ((hour - 19) / 1.8) ** 2)
    morning_peak = 3.5 * np.exp(-0.5 * ((hour - 8) / 2.0) ** 2)
    daily_load = 3.0 * np.sin(2 * np.pi * (hour - 8) / 24) + evening_peak + morning_peak
    scheduled_peak = (
        np.where(np.isin(hour, [18, 19, 20]), 10.0, 0.0)
        + np.where(np.isin(hour, [22, 23]) & (temperature > 30.0), 8.0, 0.0)
    )
    weekend_adjustment = np.where(weekend == 1, -2.0, 0.0)
    demand = (
        116
        + 0.95 * (temperature - 30)
        + daily_load
        + weekend_adjustment
        + scheduled_peak
        + rng.normal(0, 2.2, len(timestamps))
    )
    demand = np.maximum(demand, 70.0)

    demand_series = pd.Series(demand)
    lag = demand_series.shift(1)
    rolling_mean = lag.rolling(24, min_periods=1).mean()
    rolling_std = lag.rolling(24, min_periods=2).std().fillna(3.0)
    next_demand = demand_series.shift(-1)
    spike = (next_demand > 1.10 * rolling_mean).astype("int64")

    frame = pd.DataFrame({
        "timestamp": timestamps,
        "hour": hour,
        "day_of_week": dow,
        "is_weekend": weekend,
        "temperature_c": np.round(temperature, 2),
        "humidity_pct": np.round(humidity, 2),
        "current_demand_mw": np.round(demand, 2),
        "demand_lag_1h_mw": np.round(lag, 2),
        "rolling_mean_24h_mw": np.round(rolling_mean, 2),
        "rolling_std_24h_mw": np.round(rolling_std, 2),
        "spike_next_hour": spike,
    })
    frame = frame.loc[frame["timestamp"].dt.year == 2025].copy()
    frame = frame.dropna().reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    return frame


if __name__ == "__main__":
    df = generate()
    print(f"Created {len(df):,} synthetic hourly records: {OUTPUT}")
    print("Target counts:", df["spike_next_hour"].value_counts().sort_index().to_dict())
