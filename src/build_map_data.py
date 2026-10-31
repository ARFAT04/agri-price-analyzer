"""
build_map_data.py
Builds a small file of monthly average prices for the geocoded districts, so the
map page works without the large raw/clean data files.

Run from the project root (after preprocess.py and weather.py):
    python src/build_map_data.py
"""
from pathlib import Path

import pandas as pd

PROC = Path("data/processed")


def main():
    coords = pd.read_csv(PROC / "district_coords.csv")
    df = pd.read_csv(PROC / "clean_all.csv", parse_dates=["date"])
    df = df.merge(coords, on=["state", "district"], how="inner")   # only districts with coordinates
    df["month"] = df["date"].dt.to_period("M").astype(str)
    out = (df.groupby(["state", "district", "lat", "lon", "commodity", "month"], as_index=False)
             .agg(avg_price=("modal_price", "mean"), records=("modal_price", "size")))
    out["avg_price"] = out["avg_price"].round(0)
    out.to_csv(PROC / "district_prices.csv", index=False)
    print(f"Saved {len(out)} rows for {coords.shape[0]} districts -> district_prices.csv")


if __name__ == "__main__":
    main()
