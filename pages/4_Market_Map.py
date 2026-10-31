import branca.colormap as cm
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.utils import PROC, SPARSE_NOTE, show_footer

st.set_page_config(page_title="Market Map", page_icon="🗺️", layout="wide")
st.title("🗺️ Market Map")


@st.cache_data
def load_map_data():
    return pd.read_csv(PROC / "district_prices.csv")


data = load_map_data()
commodity = st.sidebar.selectbox("Commodity", sorted(data["commodity"].unique()))
months = ["Whole period"] + sorted(data["month"].unique())
month = st.sidebar.selectbox("Month", months)

sub = data[data["commodity"] == commodity]
if month != "Whole period":
    sub = sub[sub["month"] == month]
# average price per district (weighted by number of records)
sub = (sub.assign(total=sub["avg_price"] * sub["records"])
          .groupby(["state", "district", "lat", "lon"], as_index=False)
          .agg(total=("total", "sum"), records=("records", "sum")))
sub["price"] = (sub["total"] / sub["records"]).round(0)

if sub.empty:
    st.warning(f"No {commodity} prices in {month} for the mapped districts.")
    st.stop()

colormap = cm.LinearColormap(["green", "yellow", "red"], vmin=sub["price"].min(), vmax=sub["price"].max(),
                             caption="Average modal price (Rs/quintal)")
m = folium.Map(location=[sub["lat"].mean(), sub["lon"].mean()], zoom_start=6, tiles="OpenStreetMap")
for _, r in sub.iterrows():
    folium.CircleMarker(
        location=[r["lat"], r["lon"]], radius=11, color="black", weight=1,
        fill=True, fill_color=colormap(r["price"]), fill_opacity=0.85,
        tooltip=f"{r['district']}, {r['state']}: Rs {r['price']:,.0f}",
        popup=f"<b>{r['district']}</b><br>{r['state']}<br>Avg price: Rs {r['price']:,.0f}<br>Records: {int(r['records'])}",
    ).add_to(m)
colormap.add_to(m)
st_folium(m, height=520, width=None, returned_objects=[])

st.caption(f"{commodity}, {month}. Each circle is a district (the busiest districts of each state) "
           "coloured by its average modal price. Only districts with coordinates are shown.")
if (sub["state"] == "Madhya Pradesh").any():
    st.caption("Note: " + SPARSE_NOTE)
st.dataframe(sub[["state", "district", "price", "records"]].sort_values("price", ascending=False)
             .rename(columns={"price": "avg price (Rs/quintal)"}), width="stretch", hide_index=True)

show_footer()