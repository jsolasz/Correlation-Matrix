import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Multi-Asset Correlations", layout="wide")

ASSET_CLASSES = {
    "Equities": ["ES=F", "YM=F", "NQ=F", "NKD=F", "^VIX"],
    "Fixed Income": ["ZN=F", "ZB=F"],
    "Commodities": ["CL=F", "HO=F", "GC=F", "SI=F", "HG=F"],
    "Forex": [
        "EURUSD=X",
        "JPY=X",
        "GBPUSD=X",
        "AUDUSD=X",
        "NZDUSD=X",
        "USDCAD=X",
        "USDCHF=X",
        "DX-Y.NYB",
    ],
}

ALL_SYMBOLS = []
for symbols in ASSET_CLASSES.values():
    for sym in symbols:
        if sym not in ALL_SYMBOLS:
            ALL_SYMBOLS.append(sym)

LABELS = {
    "ES=F": "S&P 500",
    "YM=F": "Dow",
    "NQ=F": "Nasdaq 100",
    "NKD=F": "Nikkei 225",
    "^VIX": "VIX Index",
    "ZN=F": "10Y Note",
    "ZB=F": "30Y Bond",
    "CL=F": "Crude Oil",
    "HO=F": "Heating Oil",
    "GC=F": "Gold",
    "SI=F": "Silver",
    "HG=F": "Copper",
    "EURUSD=X": "EUR/USD",
    "JPY=X": "USD/JPY",
    "GBPUSD=X": "GBP/USD",
    "AUDUSD=X": "AUD/USD",
    "NZDUSD=X": "NZD/USD",
    "USDCAD=X": "USD/CAD",
    "USDCHF=X": "USD/CHF",
    "DX-Y.NYB": "USD (DXY Index)",
}

FREQ_CONFIG = {
    "5m": {"yf_interval": "5m", "periods": ["1d", "5d", "1mo"], "default_period": "5d"},
    "10m": {"yf_interval": "5m", "periods": ["1d", "5d", "1mo"], "default_period": "5d"},
    "30m": {"yf_interval": "30m", "periods": ["5d", "1mo", "3mo", "6mo"], "default_period": "1mo"},
    "1h": {"yf_interval": "60m", "periods": ["5d", "1mo", "3mo", "6mo", "1y"], "default_period": "3mo"},
    "1d": {"yf_interval": "1d", "periods": ["1mo", "3mo", "6mo", "1y", "2y", "5y"], "default_period": "1y"},
}


@st.cache_data(show_spinner=False)
def load_data(symbols: tuple[str, ...], lookback: str, freq: str) -> pd.DataFrame:
    yf_interval = FREQ_CONFIG[freq]["yf_interval"]
    data = yf.download(
        tickers=list(symbols),
        period=lookback,
        interval=yf_interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        close = pd.DataFrame(index=data.index)
        for sym in symbols:
            if (sym, "Close") in data.columns:
                close[sym] = data[(sym, "Close")]
    else:
        close = pd.DataFrame({symbols[0]: data["Close"]}, index=data.index)

    close = close.dropna(how="all")

    if freq == "10m":
        close = close.resample("10min").last()

    return close


def compute_returns(close_px: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "pct_change":
        rets = close_px.pct_change()
    else:
        rets = np.log(close_px / close_px.shift(1))
    return rets.replace([np.inf, -np.inf], np.nan)


def render_asset_tab(asset_name: str, class_symbols: list[str], close_px_all: pd.DataFrame, frequency: str, period: str, method: str, min_obs: int) -> None:
    tab_key = asset_name.lower().replace(" ", "_")

    universe = [sym for sym in class_symbols if sym in close_px_all.columns]
    if len(universe) < 2:
        st.warning(f"Not enough data loaded for {asset_name}.")
        return

    selected = st.multiselect(
        f"{asset_name} symbols",
        options=universe,
        default=universe,
        format_func=lambda x: f"{LABELS.get(x, x)} ({x})",
        key=f"{tab_key}_symbols",
    )

    if len(selected) < 2:
        st.warning("Select at least two symbols to compute correlations.")
        return

    rets = compute_returns(close_px_all[selected], method)
    valid_counts = rets.notna().sum()
    available = [sym for sym in selected if valid_counts.get(sym, 0) >= min_obs]

    if len(available) < 2:
        st.error("Not enough overlapping observations after filtering. Lower the minimum observations.")
        st.dataframe(valid_counts.rename("valid_obs").to_frame())
        return

    rets = rets[available].dropna(how="all")
    corr = rets.corr(method="pearson")

    rename_map = {sym: LABELS.get(sym, sym) for sym in corr.columns}
    corr_display = corr.rename(index=rename_map, columns=rename_map)

    fig = px.imshow(
        corr_display,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        text_auto=".2f",
        aspect="auto",
        title=f"{asset_name} Correlation Matrix ({frequency}, {period})",
    )
    fig.update_layout(coloraxis_colorbar_title="Corr")

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(fig, use_container_width=True, key=f"{tab_key}_matrix_chart")

    with right:
        st.subheader("Coverage")
        coverage = valid_counts.sort_values(ascending=False).rename("valid_obs").to_frame()
        coverage.index = coverage.index.map(lambda x: f"{LABELS.get(x, x)} ({x})")
        st.dataframe(coverage)


st.title("Multi-Asset Correlation Matrix")
st.caption("Color-coded matrices and rolling correlations by asset class")

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    frequency = st.selectbox("Frequency", options=list(FREQ_CONFIG.keys()), index=3)
period_options = FREQ_CONFIG[frequency]["periods"]
default_period = FREQ_CONFIG[frequency]["default_period"]
with col2:
    period = st.selectbox(
        "Lookback period",
        period_options,
        index=period_options.index(default_period),
        help="Available periods vary by selected frequency due to Yahoo intraday limits.",
    )
with col3:
    method = st.selectbox("Return type", ["pct_change", "log_return"], index=0)
with col4:
    min_obs = st.slider("Min overlapping observations", min_value=20, max_value=300, value=80)

with st.spinner(f"Downloading {frequency} data..."):
    close_px_all = load_data(tuple(ALL_SYMBOLS), period, frequency)

if close_px_all.empty:
    st.error("No data returned from yfinance for the selected inputs.")
    st.stop()

tabs = st.tabs(["All Assets", "Equities", "Fixed Income", "Commodities", "Forex"])
asset_order = ["All Assets", "Equities", "Fixed Income", "Commodities", "Forex"]
for tab, asset_name in zip(tabs, asset_order):
    with tab:
        class_symbols = ALL_SYMBOLS if asset_name == "All Assets" else ASSET_CLASSES[asset_name]
        render_asset_tab(
            asset_name=asset_name,
            class_symbols=class_symbols,
            close_px_all=close_px_all,
            frequency=frequency,
            period=period,
            method=method,
            min_obs=min_obs,
        )
