"""Shared helpers for the Streamlit app."""
from pathlib import Path

import pandas as pd
import streamlit as st

PROC = Path("data/processed")

FULL = ["Onion", "Potato"]          # full two-year history
LIMITED = ["Tomato", "Wheat"]       # partial history, shown with a warning
LIMITED_NOTE = {
    "Tomato": "Limited data: tomato prices exist only for Jun-Nov 2023 in the source dataset.",
    "Wheat": "Limited data: wheat prices exist only for Jun 2023 - early Feb 2024 in the source dataset.",
}
STATES = ["Uttar Pradesh", "Maharashtra", "Madhya Pradesh"]
SPARSE_NOTE = ("Madhya Pradesh has many days with very few markets reporting, "
               "so its daily prices are noisier and it is not used for forecasting.")
WEATHER_LABELS = {"temp_mean": "Temperature (C)", "rain_mm": "Rainfall (mm)",
                  "humidity_mean": "Humidity (%)"}


def slug(commodity: str, state: str) -> str:
    return f"{commodity.lower()}_{state.lower().replace(' ', '_')}"


@st.cache_data
def load_series(commodity: str, state: str) -> pd.DataFrame:
    """Daily price + weather for one commodity and state."""
    df = pd.read_csv(PROC / f"merged_{slug(commodity, state)}.csv", parse_dates=["date"])
    return df


@st.cache_data
def load_summary() -> pd.DataFrame:
    return pd.read_csv(PROC / "series_summary.csv")


def show_footer():
    st.divider()
    st.caption("Data: Agmarknet mandi prices (Ministry of Agriculture & Farmers Welfare, "
               "via Kaggle) | Weather: Open-Meteo")