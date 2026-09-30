"""Nifty 500 scanner for swing and positional setups. Educational only, not advice."""
import io

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from urllib.parse import quote

st.set_page_config(page_title="NSE Swing Scanner", layout="wide")
LIST_URL = "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv"


@st.cache_data(ttl=86400)
def load_universe():
    r = requests.get(LIST_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))[["Company Name", "Industry", "Symbol"]]


@st.cache_data(ttl=1800, show_spinner="Downloading prices (first run takes a minute or two)...")
def load_prices(symbols: tuple):
    out = {}
    for i in range(0, len(symbols), 100):
        tick = [s + ".NS" for s in symbols[i:i + 100]]
        d = yf.download(tick, period="1y", interval="1d", group_by="ticker",
                        auto_adjust=True, threads=True, progress=False)
        for t in tick:
            try:
                x = d[t].dropna()
            except KeyError:
                continue
            if len(x) > 210:
                out[t[:-3]] = x
    return out


@st.cache_data(ttl=1800)
def load_nifty():
    d = yf.download("^NSEI", period="1y", interval="1d", auto_adjust=True, progress=False)
    return d["Close"].squeeze().dropna()


@st.cache_data(ttl=86400)
def fundamentals(sym):
    try:
        i = yf.Ticker(sym + ".NS").info
    except Exception:
        return {}

    def g(k, m=1):
        return round(i[k] * m, 1) if i.get(k) is not None else None

    return {"Debt/Equity": g("debtToEquity"), "Profit growth %": g("earningsGrowth", 100),
            "Sales growth %": g("revenueGrowth", 100)}


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def analyse(x, nifty_ret):
    c, v = x["Close"], x["Volume"]
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    macd = ema(c, 12) - ema(c, 26)
    sig = ema(macd, 9)
    r = rsi(c).iloc[-1]
    vr = v.iloc[-1] / v.iloc[-21:-1].mean()
    last = c.iloc[-1]
    ok = {
        "Trend (price>20>50>200 EMA)": bool(last > e20.iloc[-1] > e50.iloc[-1] > e200.iloc[-1]),
        "RSI 55-80": bool(55 <= r <= 80),
        "MACD up, above zero": bool(macd.iloc[-1] > sig.iloc[-1] and macd.iloc[-1] > 0),
        "Volume 1.5x avg": bool(vr >= 1.5),
        "Close above 20-day high": bool(last > c.iloc[-21:-1].max()),
        "Within 10% of 52w high": bool(last >= 0.9 * c.max()),
        "Beats Nifty (3 months)": bool(last / c.iloc[-64] - 1 > nifty_ret),
        "Tight base (prior 10d range under 8%)": bool(
            (x["High"].iloc[-11:-1].max() - x["Low"].iloc[-11:-1].min()) / last <= 0.08),
        "Strong candle (close in top 25%)": bool(
            last >= x["Low"].iloc[-1] + 0.75 * (x["High"].iloc[-1] - x["Low"].iloc[-1])),
    }
    stop = x["Low"].iloc[-10:].min()
    return {
        "Close": round(last, 2), "Score": int(sum(ok.values())), "RSI": round(r, 1),
        "Volume x": round(vr, 1), "Stop (10d low)": round(stop, 2),
        "Risk %": round((last - stop) / last * 100, 1),
        "Liquidity (Rs cr)": round((c * v).iloc[-20:].mean() / 1e7, 1),
        **{k: ("✔" if b else "") for k, b in ok.items()},
    }


def draw(x, sym):
    c = x["Close"]
    d = pd.DataFrame({"e20": ema(c, 20), "e50": ema(c, 50), "e200": ema(c, 200),
                      "rsi": rsi(c), "vavg": x["Volume"].rolling(20).mean()}).join(x).tail(130)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[.6, .2, .2], vertical_spacing=.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                                 name="Price", increasing_line_color="#1f8a5b",
                                 decreasing_line_color="#c2413b"), 1, 1)
    for k, name, col in (("e20", "EMA 20", "#3b82f6"), ("e50", "EMA 50", "#d98a00"), ("e200", "EMA 200", "#888888")):
        fig.add_trace(go.Scatter(x=d.index, y=d[k], name=name, line=dict(color=col, width=1.4)), 1, 1)
    fig.add_hline(y=c.iloc[-21:-1].max(), line_dash="dash", line_color="#d98a00",
                  annotation_text="20-day high (breakout level)", row=1, col=1)
    fig.add_hline(y=x["Low"].iloc[-10:].min(), line_dash="dash", line_color="#c2413b",
                  annotation_text="Stop (10-day low)", row=1, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="Volume", marker_color="#8aa0ab"), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["vavg"], name="20-day avg volume",
                             line=dict(color="#d98a00", width=1.2)), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["rsi"], name="RSI 14", line=dict(color="#3b82f6", width=1.4)), 3, 1)
    for lvl in (40, 80):
        fig.add_hline(y=lvl, line_dash="dot", line_color="#888888", row=3, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(height=680, title=sym, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    return fig


st.title("Nifty 500 scanner for swing and positional setups")
st.caption("Prices: Yahoo Finance (unofficial, about 15 minutes delayed). Educational only, not financial "
           "advice. A high score is a shortlist, not a prediction.")

try:
    uni = load_universe()
except Exception:
    up = st.file_uploader("Could not fetch the Nifty 500 list. Upload ind_nifty500list.csv "
                          "from niftyindices.com", type="csv")
    if up is None:
        st.stop()
    uni = pd.read_csv(up)[["Company Name", "Industry", "Symbol"]]

nifty = load_nifty()
(st.success if nifty.iloc[-1] > nifty.rolling(50).mean().iloc[-1] and
 nifty.iloc[-1] > nifty.rolling(200).mean().iloc[-1] else st.warning)(
    "Market check: Nifty is above its 50 and 200 DMA, so conditions favour breakouts."
    if nifty.iloc[-1] > nifty.rolling(50).mean().iloc[-1] and nifty.iloc[-1] > nifty.rolling(200).mean().iloc[-1]
    else "Market check: Nifty is below its 50 or 200 DMA. Trade smaller or wait.")

min_score = st.sidebar.slider("Minimum score (out of 9)", 0, 9, 7)
min_liq = st.sidebar.number_input("Minimum daily traded value (Rs crore)", 0.0, 500.0, 5.0)
sectors = st.sidebar.multiselect("Industry filter (optional)", sorted(uni["Industry"].unique()))

if st.button("Run scan", type="primary"):
    prices = load_prices(tuple(uni["Symbol"]))
    ret = nifty.iloc[-1] / nifty.iloc[-64] - 1
    rows = [{"Symbol": s, **analyse(x, ret)} for s, x in prices.items()]
    st.session_state["res"] = pd.DataFrame(rows).merge(uni, on="Symbol")

res = st.session_state.get("res")
if res is not None:
    df = res[(res["Score"] >= min_score) & (res["Liquidity (Rs cr)"] >= min_liq)]
    if sectors:
        df = df[df["Industry"].isin(sectors)]
    df = df.sort_values(["Score", "Volume x"], ascending=False)
    st.subheader(f"{len(df)} stocks passed out of {len(res)} scanned")
    if st.sidebar.checkbox("Add fundamentals for top 40 (slow, optional)"):
        df = df.head(40)
        df = df.join(pd.DataFrame([fundamentals(s) for s in df["Symbol"]], index=df.index))
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False), "scan.csv")
    if len(df):
        sym = st.selectbox("Open chart for", df["Symbol"].tolist())
        row = df[df["Symbol"] == sym].iloc[0]
        px = load_prices(tuple(uni["Symbol"])).get(sym)
        if px is not None:
            st.plotly_chart(draw(px, sym), use_container_width=True)
        names = [k for k in row.index if row[k] in ("✔", "")]
        st.success("Passed: " + (", ".join(k for k in names if row[k] == "✔") or "none"))
        st.warning("Not passed: " + (", ".join(k for k in names if row[k] == "") or "none"))
        st.link_button("Also open on TradingView (optional)",
                       "https://in.tradingview.com/chart/?symbol=NSE%3A" + quote(sym, safe=""))
    st.info("Next step: pick a stock above, study its chart and apply your Day 30 checklist "
            "(tight base, clean breakout candle, stop and 1:2 reward). Skip it if any check fails.")
else:
    st.write("Press **Run scan** to analyse all Nifty 500 stocks.")
