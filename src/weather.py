"""
weather.py
1) Finds coordinates for the busiest districts of each state (Open-Meteo geocoding)
2) Downloads daily weather history for them (Open-Meteo archive API, no key needed)
3) Averages weather per state and merges it with each price series

Run from the project root (after preprocess.py):
    python src/weather.py
"""
import time
from pathlib import Path

import pandas as pd
import requests

PROC = Path("data/processed")
START, END = "2023-06-06", "2025-06-11"     # same range as the price data
TOP_N = 5                                   # districts used per state

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = ["temperature_2m_mean", "precipitation_sum", "relative_humidity_2m_mean"]


def pick_districts() -> pd.DataFrame:
    """Top districts per state, ranked by number of price records."""
    m = pd.read_csv(PROC / "markets.csv")
    d = m.groupby(["state", "district"], as_index=False)["records"].sum()
    d = d.sort_values(["state", "records"], ascending=[True, False])
    return d.groupby("state").head(TOP_N).reset_index(drop=True)


def geocode(district: str, state: str):
    """Return (lat, lon) for a district, or None if not found."""
    r = requests.get(GEO_URL, params={"name": district, "count": 10, "language": "en",
                                      "format": "json", "countryCode": "IN"}, timeout=30)
    r.raise_for_status()
    for res in r.json().get("results", []):
        if str(res.get("admin1", "")).lower() == state.lower():
            return res["latitude"], res["longitude"]
    return None


def get_coords() -> pd.DataFrame:
    cache = PROC / "district_coords.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    for _, r in pick_districts().iterrows():
        try:
            pt = geocode(r["district"], r["state"])
        except requests.RequestException as e:
            print(f"  geocoding failed for {r['district']}: {e}")
            pt = None
        if pt is None:
            print(f"  not found: {r['district']}, {r['state']}")
        else:
            rows.append({"state": r["state"], "district": r["district"], "lat": pt[0], "lon": pt[1]})
        time.sleep(0.5)
    coords = pd.DataFrame(rows)
    coords.to_csv(cache, index=False)
    return coords


def fetch_weather(lat: float, lon: float) -> pd.DataFrame:
    params = {"latitude": lat, "longitude": lon, "start_date": START, "end_date": END,
              "daily": ",".join(DAILY_VARS), "timezone": "Asia/Kolkata"}
    r = requests.get(ARCHIVE_URL, params=params, timeout=60)
    if r.status_code == 400:                    # fall back if a variable is not offered
        params["daily"] = "temperature_2m_mean,precipitation_sum"
        r = requests.get(ARCHIVE_URL, params=params, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def state_weather(coords: pd.DataFrame) -> dict:
    """Average the daily weather of a state's districts into one series."""
    out = {}
    for state, grp in coords.groupby("state"):
        frames = []
        for _, r in grp.iterrows():
            frames.append(fetch_weather(r["lat"], r["lon"]))
            time.sleep(1)
        w = pd.concat(frames).groupby("date", as_index=False).mean(numeric_only=True)
        w = w.rename(columns={"temperature_2m_mean": "temp_mean",
                              "precipitation_sum": "rain_mm",
                              "relative_humidity_2m_mean": "humidity_mean"})
        w.to_csv(PROC / f"weather_{state.lower().replace(' ', '_')}.csv", index=False)
        out[state] = w
        print(f"  weather saved for {state} ({len(grp)} districts)")
    return out


def merge_all(weather: dict):
    summ = pd.read_csv(PROC / "series_summary.csv")
    for _, s in summ.iterrows():
        slug = f"{s['commodity'].lower()}_{s['state'].lower().replace(' ', '_')}"
        prices = pd.read_csv(PROC / f"series_{slug}.csv", parse_dates=["date"])
        merged = prices.merge(weather[s["state"]], on="date", how="left")
        merged.to_csv(PROC / f"merged_{slug}.csv", index=False)
    print(f"Merged {len(summ)} series with weather.")


def main():
    print("Finding district coordinates...")
    coords = get_coords()
    print(coords.to_string(index=False))
    print("Downloading weather...")
    weather = state_weather(coords)
    merge_all(weather)


if __name__ == "__main__":
    main()