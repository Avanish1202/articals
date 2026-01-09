# ==================================================
# IMPORTS
# ==================================================
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI

# ==================================================
# OPENAI CLIENT (API KEY INLINE)
# ==================================================
client = OpenAI(
    api_key="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"
)

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Smart Candlestick + AI Report",
    layout="wide"
)

st.title("📊 Smart Candlestick + Technical Analysis Dashboard")

# ==================================================
# SIDEBAR CONTROLS
# ==================================================
st.sidebar.header("⚙️ Chart Controls")

symbol = st.sidebar.text_input("Stock Symbol", "RELIANCE.NS")

chart_type = st.sidebar.radio("Chart Type", ["Daily", "Intraday"])

if chart_type == "Daily":
    interval = st.sidebar.selectbox("Interval", ["1d", "1wk"], index=0)
    period = st.sidebar.selectbox("Period", ["3mo", "6mo", "1y"], index=1)
else:
    interval = st.sidebar.selectbox("Intraday Interval", ["1m", "5m", "15m", "30m"], index=2)
    period = "7d" if interval in ["1m", "5m"] else "60d"

show_candle_names = st.sidebar.checkbox("Show Candle Names", True)
label_density = st.sidebar.slider("Candle Label Density", 1, 20, 5)

st.sidebar.subheader("📈 Technical Indicators")
selected_indicators = st.sidebar.multiselect(
    "Select Indicators",
    ["EMA 20", "EMA 50", "EMA 200", "Support", "Resistance", "Volume", "MACD", "RSI"],
    default=["EMA 20", "EMA 50", "EMA 200", "Support", "Resistance", "Volume"]
)

st.sidebar.markdown("---")
generate_report = st.sidebar.button("🧠 Generate AI Report")
refresh_data = st.sidebar.button("🔄 Refresh Data")

# ==================================================
# SESSION STATE
# ==================================================
if "data_key" not in st.session_state:
    st.session_state.data_key = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "final_df" not in st.session_state:
    st.session_state.final_df = None

current_key = f"{symbol}_{interval}_{period}"

# ==================================================
# DATA LOADING
# ==================================================
def load_price_data(symbol, interval, period):
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna()
    return df

if refresh_data or st.session_state.data_key != current_key or st.session_state.raw_df is None:
    with st.spinner("📥 Loading market data..."):
        df = load_price_data(symbol, interval, period)

        if df.empty:
            st.error("❌ No data available")
            st.stop()

        if "m" in interval:
            df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index
            df.index = df.index.tz_convert("Asia/Kolkata")
        else:
            df.index = df.index.tz_localize(None)

        st.session_state.raw_df = df
        st.session_state.data_key = current_key
        st.session_state.final_df = None

df = st.session_state.raw_df

# ==================================================
# INDICATORS
# ==================================================
def calculate_indicators(df):
    df = df.copy()

    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

    df["Support"] = df["Low"].rolling(20).min()
    df["Resistance"] = df["High"].rolling(20).max()

    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    df["VOL_MA"] = df["Volume"].rolling(20).mean()
    df["VOLUME_SPIKE"] = df["Volume"] > 1.5 * df["VOL_MA"]

    return df

# ==================================================
# CANDLE CLASSIFICATION
# ==================================================
def classify_candles(df):
    def candle_name(r):
        o, h, l, c = r["Open"], r["High"], r["Low"], r["Close"]
        body = abs(c - o)
        rng = h - l
        upper = h - max(o, c)
        lower = min(o, c) - l

        if rng == 0:
            return "Flat"
        if body / rng <= 0.1:
            return "Doji"
        if upper < rng * 0.05 and lower < rng * 0.05:
            return "Bullish Marubozu" if c > o else "Bearish Marubozu"
        if lower >= 2 * body:
            return "Hammer" if c > o else "Hanging Man"
        if upper >= 2 * body:
            return "Inverted Hammer" if c > o else "Shooting Star"
        if body / rng >= 0.6:
            return "Strong Bullish" if c > o else "Strong Bearish"
        return "Spinning Top"

    df = df.copy()
    df["CANDLE"] = df.apply(candle_name, axis=1)
    return df

if st.session_state.final_df is None:
    st.session_state.final_df = classify_candles(calculate_indicators(df))

df = st.session_state.final_df

# ==================================================
# TECHNICAL SUMMARY
# ==================================================
last = df.iloc[-1]

trend = "Sideways / Range"
if last["EMA_20"] > last["EMA_50"] > last["EMA_200"]:
    trend = "Strong Uptrend"
elif last["EMA_20"] < last["EMA_50"] < last["EMA_200"]:
    trend = "Strong Downtrend"

momentum = "Neutral Momentum"
if last["RSI"] >= 60:
    momentum = "Bullish Momentum"
elif last["RSI"] <= 40:
    momentum = "Bearish Momentum"

buy_zone = trend == "Strong Uptrend" and momentum == "Bullish Momentum" and last["VOLUME_SPIKE"]
sell_zone = trend == "Strong Downtrend" and momentum == "Bearish Momentum" and last["VOLUME_SPIKE"]

st.subheader("🧠 Technical Analysis Summary")
st.markdown(f"""
**Trend:** `{trend}`  
**Momentum:** `{momentum}`  
**RSI:** `{round(last['RSI'], 2)}`  
**Volume Spike:** `{"Yes" if last['VOLUME_SPIKE'] else "No"}`  
**Candle Pattern:** `{last['CANDLE']}`  
**Support:** `{round(last['Support'],2)}`  
**Resistance:** `{round(last['Resistance'],2)}`
""")

if buy_zone:
    st.success("📈 BUY ZONE – High probability setup")
elif sell_zone:
    st.error("📉 SELL ZONE – Bearish confirmation")
else:
    st.warning("⏸ WAIT ZONE – No clear edge")

# ==================================================
# CHART
# ==================================================
rows = 1 + ("Volume" in selected_indicators) + ("MACD" in selected_indicators) + ("RSI" in selected_indicators)
fig = make_subplots(rows=rows, cols=1, shared_xaxes=True)

r = 1
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Price"
), row=r, col=1)

for name, col in [("EMA 20", "EMA_20"), ("EMA 50", "EMA_50"), ("EMA 200", "EMA_200")]:
    if name in selected_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=name), row=r, col=1)

if show_candle_names:
    for i, row_ in df.iloc[::label_density].iterrows():
        fig.add_annotation(
            x=i,
            y=row_["High"] * 1.01,
            text=row_["CANDLE"],
            showarrow=False,
            textangle=90,
            font=dict(size=9, color="yellow")
        )

fig.update_layout(
    template="plotly_dark",
    height=900,
    hovermode="x unified",
    title=f"{symbol} ({interval})"
)

st.plotly_chart(fig, use_container_width=True)

# ==================================================
# AI REPORT
# ==================================================
if generate_report:
    with st.spinner("Generating AI Technical Report..."):
        prompt = f"""
        You are a professional stock market analyst.

        Stock: {symbol}
        Close Price: {last['Close']}
        EMA 20: {last['EMA_20']}
        EMA 50: {last['EMA_50']}
        EMA 200: {last['EMA_200']}
        RSI: {last['RSI']}
        MACD: {last['MACD']}

        Provide:
        1. Trend analysis
        2. Momentum analysis
        3. Overbought/Oversold condition
        4. Short-term outlook
        5. Buy / Hold / Sell suggestion
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a stock market expert."},
                {"role": "user", "content": prompt}
            ]
        )

        st.subheader("🧠 AI Technical Analysis Report")
        st.write(response.choices[0].message.content)
