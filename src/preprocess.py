"""
preprocess.py
Cleans the raw Agmarknet mandi price file and builds one daily price series
per (commodity, state).

Run from the project root:
    python src/preprocess.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw/Agriculture_price_dataset.csv")
OUT_DIR = Path("data/processed")

STATES = ["Uttar Pradesh", "Maharashtra", "Madhya Pradesh"]
FULL_COMMODITIES = ["Onion", "Potato"]      # full two-year history
LIMITED_COMMODITIES = ["Tomato", "Wheat"]   # partial history, kept separate

# Fix spelling variants of state names found in the raw file
STATE_FIX = {
    "Tamilnadu": "Tamil Nadu",
    "Chattisgarh": "Chhattisgarh",
    "Orissa": "Odisha",
    "Uttrakhand": "Uttarakhand",
    "Gao": "Goa",
    "Jammu & Kashmir": "Jammu and Kashmir",
}


def load_and_clean(path: Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Consistent column names
    df = df.rename(columns={
        "STATE": "state",
        "District Name": "district",
        "Market Name": "market",
        "Commodity": "commodity",
        "Variety": "variety",
        "Grade": "grade",
        "Min_Price": "min_price",
        "Max_Price": "max_price",
        "Modal_Price": "modal_price",
        "Price Date": "date",
    })

    # Dates are month/day/year (e.g. 6/6/2023)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")

    # Text clean-up: trim spaces, fix state spellings, tidy district names
    df["state"] = df["state"].str.strip().replace(STATE_FIX)
    df["district"] = df["district"].str.strip().str.title()
    df["market"] = df["market"].str.strip()
    df["commodity"] = df["commodity"].str.strip().str.title()

    # Remove rows with impossible prices (zero or max < min)
    df = df[(df["modal_price"] > 0) & (df["min_price"] > 0) & (df["max_price"] >= df["min_price"])]

    # Remove extreme outliers per commodity (data-entry errors such as Rs 460000/quintal).
    # Keep prices within 1st-99.5th percentile of that commodity.
    lo = df.groupby("commodity")["modal_price"].transform(lambda s: s.quantile(0.01))
    hi = df.groupby("commodity")["modal_price"].transform(lambda s: s.quantile(0.995))
    df = df[(df["modal_price"] >= lo) & (df["modal_price"] <= hi)]

    return df.drop_duplicates().reset_index(drop=True)


def daily_series(df: pd.DataFrame, commodity: str, state: str) -> pd.DataFrame:
    """One row per day: median modal price across all markets in the state."""
    sub = df[(df["commodity"] == commodity) & (df["state"] == state)]
    daily = (
        sub.groupby("date")
        .agg(modal_price=("modal_price", "median"),
             min_price=("min_price", "median"),
             max_price=("max_price", "median"),
             n_markets=("market", "nunique"))
        .sort_index()
    )
    # Fill missing calendar days: interpolate short gaps only (max 3 days)
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index)
    daily["was_missing"] = daily["modal_price"].isna()
    for col in ["modal_price", "min_price", "max_price"]:
        daily[col] = daily[col].interpolate(limit=3, limit_area="inside")
    daily.index.name = "date"
    return daily.reset_index()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_clean()
    df.to_csv(OUT_DIR / "clean_all.csv", index=False)
    print(f"Clean rows: {len(df):,}")

    # Market-level table for the Folium map and weather lookup
    markets = (df[df["state"].isin(STATES)]
               .groupby(["state", "district", "market"]).size()
               .reset_index(name="records"))
    markets.to_csv(OUT_DIR / "markets.csv", index=False)

    summary = []
    for group, commodities in [("full", FULL_COMMODITIES), ("limited", LIMITED_COMMODITIES)]:
        for c in commodities:
            for s in STATES:
                ts = daily_series(df, c, s)
                if ts.empty:
                    continue
                ts.to_csv(OUT_DIR / f"series_{c.lower()}_{s.lower().replace(' ', '_')}.csv", index=False)
                summary.append({
                    "group": group, "commodity": c, "state": s,
                    "start": ts["date"].min().date(), "end": ts["date"].max().date(),
                    "days": len(ts), "days_missing": int(ts["was_missing"].sum()),
                    "still_nan": int(ts["modal_price"].isna().sum()),
                })
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT_DIR / "series_summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()