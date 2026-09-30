"""Nifty 500 scanner for swing and positional setups. Educational only, not advice."""
import io

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

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
    st.info("Next step: open each name on TradingView and apply your Day 30 checklist "
            "(tight base, clean breakout candle, stop and 1:2 reward). Skip it if any check fails.")
else:
    st.write("Press **Run scan** to analyse all Nifty 500 stocks.")
