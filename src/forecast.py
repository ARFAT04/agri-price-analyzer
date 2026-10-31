"""
forecast.py
Feature building, model training and evaluation for price forecasting.

Idea (kept simple on purpose):
  * For each day t we look at recent prices and weather and predict the price
    `horizon` days later.
  * The model predicts the percentage CHANGE in price between day t and day t+h
    (as a log ratio); the forecast is today's price times that change. Features are
    also ratios, so the model does not depend on the price level.
  * Rows that touch a reporting gap are dropped, so the model never learns
    across missing periods.
  * Train/test split is by time (the last 20% of rows are the test set).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

PRICE_FEATURES = ["ret1", "ret2", "ret7", "ret14", "mean7_ratio", "vol7", "month"]
WEATHER_FEATURES = ["temp", "humidity", "rain7", "rain7_lag7", "rain7_lag14"]
BASELINES = ["Naive (today's price)", "7-day average"]
MODELS = ["Ridge", "Random Forest", "XGBoost"]


def build_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """One row per day: features known on that day + the price `horizon` days later."""
    d = df.set_index("date").asfreq("D")
    p = d["modal_price"]
    X = pd.DataFrame(index=d.index)
    for k in (0, 1, 2, 7, 14):
        X[f"price_lag{k}"] = p.shift(k)
    X["roll_mean7"] = p.rolling(7, min_periods=7).mean()
    # scale-free versions (ratios to today's price) so the model works at any price level
    for k in (1, 2, 7, 14):
        X[f"ret{k}"] = p / p.shift(k) - 1
    X["mean7_ratio"] = X["roll_mean7"] / p - 1
    X["vol7"] = p.rolling(7, min_periods=7).std() / X["roll_mean7"]
    X["month"] = d.index.month
    if "temp_mean" in d.columns:
        X["temp"] = d["temp_mean"]
        X["humidity"] = d["humidity_mean"] if "humidity_mean" in d.columns else np.nan
        X["rain7"] = d["rain_mm"].rolling(7, min_periods=7).sum()
        X["rain7_lag7"] = X["rain7"].shift(7)
        X["rain7_lag14"] = X["rain7"].shift(14)
    X["target_price"] = p.shift(-horizon)
    return X


def _metrics(actual, pred) -> dict:
    actual, pred = np.asarray(actual), np.asarray(pred)
    return {"MAE": mean_absolute_error(actual, pred),
            "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
            "MAPE %": float(np.mean(np.abs((actual - pred) / actual)) * 100)}


def _make_model(name: str):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=3,
                                     random_state=42, n_jobs=-1)
    return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                        subsample=0.8, random_state=42, verbosity=0)


def run_forecast(df: pd.DataFrame, horizon: int = 7, test_frac: float = 0.2) -> dict:
    X_all = build_features(df, horizon)
    feats_w = [c for c in PRICE_FEATURES + WEATHER_FEATURES if c in X_all.columns]
    feats_p = PRICE_FEATURES

    usable = X_all.dropna(subset=feats_w + ["target_price"])
    if len(usable) < 60:
        return {"error": f"Only {len(usable)} usable days after removing gaps - too few to model."}

    n_test = max(int(len(usable) * test_frac), 20)
    test = usable.iloc[-n_test:]
    # drop training rows whose target date reaches into the test period (no leakage)
    train = usable[usable.index + pd.Timedelta(days=horizon) < test.index[0]]

    y_train = np.log(train["target_price"] / train["price_lag0"])
    actual = test["target_price"].values

    preds = {"Naive (today's price)": test["price_lag0"].values,
             "7-day average": test["roll_mean7"].values}
    fitted = {}
    for name in MODELS:
        m = _make_model(name).fit(train[feats_w], y_train)
        fitted[name] = m
        preds[name] = test["price_lag0"].values * np.exp(m.predict(test[feats_w]))

    # does weather help? same rows, XGBoost with and without weather columns
    m_nw = _make_model("XGBoost").fit(train[feats_p], y_train)
    pred_nw = test["price_lag0"].values * np.exp(m_nw.predict(test[feats_p]))
    ablation = pd.DataFrame({
        "XGBoost without weather": _metrics(actual, pred_nw),
        "XGBoost with weather": _metrics(actual, preds["XGBoost"]),
    }).T

    table = pd.DataFrame({k: _metrics(actual, v) for k, v in preds.items()}).T
    best = table.loc[MODELS, "RMSE"].idxmin()

    # refit the best model on all usable rows and forecast from the latest day
    latest = X_all.dropna(subset=feats_w).iloc[[-1]]
    final = _make_model(best).fit(usable[feats_w], np.log(usable["target_price"] / usable["price_lag0"]))
    forecast_price = float(latest["price_lag0"].iloc[0] * np.exp(final.predict(latest[feats_w])[0]))

    importance = None
    if hasattr(fitted["XGBoost"], "feature_importances_"):
        importance = pd.Series(fitted["XGBoost"].feature_importances_, index=feats_w).sort_values()

    return {
        "table": table.round(1), "best": best, "ablation": ablation.round(1),
        "importance": importance,
        "test_dates": test.index + pd.Timedelta(days=horizon),   # date the price refers to
        "actual": actual, "preds": preds,
        "n_train": len(train), "n_test": len(test),
        "train_range": (train.index.min().date(), train.index.max().date()),
        "test_range": (test.index.min().date(), test.index.max().date()),
        "latest_date": latest.index[0], "latest_price": float(latest["price_lag0"].iloc[0]),
        "forecast_price": forecast_price, "horizon": horizon,
    }