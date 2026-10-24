"""Agricultural Market Price Analyzer - home page.
Run with:  streamlit run app.py
"""
import streamlit as st

from src.utils import load_summary, show_footer, LIMITED_NOTE, SPARSE_NOTE

st.set_page_config(page_title="Agri Market Price Analyzer", page_icon="🌾", layout="wide")

st.title("🌾 Agricultural Market Price Analyzer")
st.write(
    "Explore daily mandi (wholesale market) prices of onion, potato, tomato and wheat "
    "in Uttar Pradesh, Maharashtra and Madhya Pradesh, and see how weather relates to price."
)

summary = load_summary()
c1, c2, c3 = st.columns(3)
c1.metric("Price series", len(summary))
c2.metric("Period covered", "Jun 2023 - Jun 2025")
c3.metric("Commodities", summary["commodity"].nunique())

st.subheader("Data coverage")
summary["short gaps filled"] = summary["days_missing"] - summary["still_nan"]
st.dataframe(
    summary.rename(columns={"still_nan": "days with no data"})[
        ["commodity", "state", "start", "end", "days", "short gaps filled", "days with no data"]
    ],
    width="stretch", hide_index=True,
)
st.info("Use the pages in the sidebar: **Price Trends** and **Weather Correlation**. "
        "Forecasting and the market map are added next.")
with st.expander("Data notes"):
    for note in LIMITED_NOTE.values():
        st.write("- " + note)
    st.write("- " + SPARSE_NOTE)
    st.write("- Onion has two reporting gaps (Jun-Jul 2024 and Dec 2024-Feb 2025). "
             "These are left empty, not filled with invented prices.")

show_footer()