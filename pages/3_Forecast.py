import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.forecast import MODELS, run_forecast
from src.utils import FULL, load_series, show_footer

st.set_page_config(page_title="Forecast", page_icon="🔮", layout="wide")
st.title("🔮 Price Forecast")
st.caption("Forecasting is done for onion and potato in Uttar Pradesh and Maharashtra, "
           "the series with enough reliable data. Madhya Pradesh, tomato and wheat are excluded.")

commodity = st.sidebar.selectbox("Commodity", FULL)
state = st.sidebar.selectbox("State", ["Uttar Pradesh", "Maharashtra"])
horizon = st.sidebar.select_slider("Forecast horizon (days ahead)", options=[1, 3, 7, 14], value=7)


@st.cache_data(show_spinner="Training models...")
def get_results(commodity, state, horizon):
    return run_forecast(load_series(commodity, state), horizon)


res = get_results(commodity, state, horizon)
if "error" in res:
    st.error(res["error"])
    st.stop()

# ---- headline numbers
f_date = (res["latest_date"] + pd.Timedelta(days=res["horizon"])).date()
change = (res["forecast_price"] / res["latest_price"] - 1) * 100
c1, c2, c3 = st.columns(3)
c1.metric(f"Last known price ({res['latest_date'].date()})", f"Rs {res['latest_price']:,.0f}")
c2.metric(f"Forecast for {f_date} ({res['best']})", f"Rs {res['forecast_price']:,.0f}", f"{change:+.1f}%")
c3.metric("Train / test days", f"{res['n_train']} / {res['n_test']}")
st.caption(f"Training period: {res['train_range'][0]} to {res['train_range'][1]}  |  "
           f"Test period (unseen by the models): {res['test_range'][0]} to {res['test_range'][1]}. "
           "The data ends in June 2025, so this is a forecast for the end of the dataset, not live prices.")

# ---- model comparison
st.subheader("Model comparison on the test period")
table = res["table"]
st.dataframe(table.style.highlight_min(subset=["MAE", "RMSE", "MAPE %"], color="#c6efce"),
             width="stretch")
st.caption("MAE and RMSE are in Rs per quintal; MAPE is the average % error. Lower is better. "
           "Naive = assume the price stays the same as today.")

best_ml = res["best"]
naive_rmse = table.loc["Naive (today's price)", "RMSE"]
if table.loc[best_ml, "RMSE"] < naive_rmse:
    gain = (1 - table.loc[best_ml, "RMSE"] / naive_rmse) * 100
    st.success(f"{best_ml} beats the naive baseline by {gain:.0f}% (RMSE).")
else:
    st.warning(f"No model beats the naive baseline on this series (best model: {best_ml}). "
               "Short-term mandi prices are hard to predict, and this is a fair result to report.")

# ---- chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=res["test_dates"], y=res["actual"], name="Actual", line=dict(color="black")))
fig.add_trace(go.Scatter(x=res["test_dates"], y=res["preds"][best_ml], name=best_ml))
fig.add_trace(go.Scatter(x=res["test_dates"], y=res["preds"]["Naive (today's price)"],
                         name="Naive baseline", line=dict(dash="dot")))
fig.update_layout(title=f"{commodity}, {state}: actual vs forecast ({horizon} days ahead)",
                  xaxis_title="Date", yaxis_title="Rs per quintal", hovermode="x unified", height=460)
st.plotly_chart(fig, width="stretch")

# ---- does weather help?
st.subheader("Does weather help?")
st.dataframe(res["ablation"], width="stretch")
w, nw = res["ablation"].loc["XGBoost with weather", "RMSE"], res["ablation"].loc["XGBoost without weather", "RMSE"]
if w < nw:
    st.write(f"Adding weather features lowered XGBoost's RMSE by {(1 - w / nw) * 100:.0f}% on this series.")
else:
    st.write("Adding weather features did not improve XGBoost on this series; "
             "price history alone was as good or better.")

# ---- feature importance
if res["importance"] is not None:
    st.subheader("What the XGBoost model relies on")
    imp = res["importance"]
    figi = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h"))
    figi.update_layout(height=360, xaxis_title="Importance")
    st.plotly_chart(figi, width="stretch")

with st.expander("How the forecast works"):
    st.markdown(f"""
- **Task:** predict the modal price {horizon} days ahead from recent prices and weather.
- **Features:** price changes over the last 1, 2, 7 and 14 days, 7-day average and volatility, month, temperature, humidity and rainfall (7-day total, also 7 and 14 days earlier).
- **Target:** the percentage change between today and {horizon} days later.
- **Models:** two simple baselines (naive and 7-day average), then Ridge regression, Random Forest and XGBoost.
- **Validation:** time-based split, the last 20% of days are the test set. Training rows whose target falls in the test period are removed, so nothing leaks.
- **Gaps:** days that touch a reporting gap are dropped, so nothing is learned across missing periods.
- **Headline forecast:** the best model (lowest RMSE) is retrained on all usable days.
""")

show_footer()