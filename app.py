"""Large + mid cap NSE scanner for swing and positional setups. Educational only, not advice."""
import io
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="NSE Swing Scanner", layout="wide")
BASE = "https://niftyindices.com/IndexConstituent/"
LISTS = {"Large": "ind_nifty100list.csv", "Mid": "ind_niftymidcap150list.csv"}


def naive(x):
    if x.index.tz is not None:
        x.index = x.index.tz_localize(None)
    return x


@st.cache_data(ttl=86400)
def load_universe():
    parts = []
    for cap, f in LISTS.items():
        r = requests.get(BASE + f, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text))[["Company Name", "Industry", "Symbol"]]
        parts.append(d.assign(Cap=cap))
    return pd.concat(parts).drop_duplicates("Symbol")


@st.cache_data(ttl=1800, show_spinner="Downloading 5 years of prices (first run takes a minute or two)...")
def load_prices(symbols: tuple):
    out = {}
    for i in range(0, len(symbols), 100):
        tick = [s + ".NS" for s in symbols[i:i + 100]]
        d = yf.download(tick, period="5y", interval="1d", group_by="ticker",
                        auto_adjust=True, threads=True, progress=False)
        for t in tick:
            try:
                x = d[t].dropna()
            except KeyError:
                continue
            if len(x) > 260:
                out[t[:-3]] = naive(x)
    return out


@st.cache_data(ttl=1800)
def load_nifty():
    d = yf.download("^NSEI", period="5y", interval="1d", auto_adjust=True, progress=False)
    return naive(d["Close"].squeeze().dropna().to_frame("Nifty 50"))["Nifty 50"]


@st.cache_data(ttl=86400)
def fundamentals(sym):
    try:
        i = yf.Ticker(sym + ".NS").info
    except Exception:
        return {}

    def g(k, m=1):
        return round(i[k] * m, 1) if i.get(k) is not None else None

    return {"P/E": g("trailingPE"), "Debt/Equity": g("debtToEquity"),
            "Profit growth %": g("earningsGrowth", 100)}


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def features(x, nifty):
    """One row per day, using exactly the same rules for today's scan and for the backtest."""
    c, v, h, l = x["Close"], x["Volume"], x["High"], x["Low"]
    n = nifty.reindex(x.index).ffill()
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    macd = ema(c, 12) - ema(c, 26)
    f = pd.DataFrame(index=x.index)
    f["Close"], f["RSI"] = c, rsi(c)
    f["Level"] = c.shift(1).rolling(60).max()                  # highest close of the prior ~3 months
    f["gap"] = (c / f["Level"] - 1) * 100
    f["fresh"] = c.shift(1) <= f["Level"].shift(1)             # yesterday was still below the level
    f["vr"] = v / v.shift(1).rolling(50).mean()
    f["rng"] = (h.shift(1).rolling(15).max() - l.shift(1).rolling(15).min()) / c * 100
    f["stop"] = l.rolling(10).min()
    f["risk"] = (c - f["stop"]) / c * 100
    f["ext"] = (c / e20 - 1) * 100
    f["strong"] = c >= l + 0.75 * (h - l)
    f["good"] = (n > n.rolling(50).mean()) & (n > n.rolling(200).mean())
    ok = pd.DataFrame({
        "Uptrend (price>50>200 EMA, 200 rising)": (c > e50) & (e50 > e200) & (e200 > e200.shift(20)),
        "RSI 50-80": f["RSI"].between(50, 80),
        "MACD up and above 0": (macd > ema(macd, 9)) & (macd > 0),
        "At or within 3% of 3-month high": f["gap"] >= -3,
        "Within 10% of 52w high": c >= 0.9 * c.rolling(252).max(),
        "Beats Nifty (3m)": c / c.shift(63) > n / n.shift(63),
        "Beats Nifty (6m)": c / c.shift(126) > n / n.shift(126),
        "Tight 3-week base (range under 10%)": f["rng"] <= 10,
    })
    f["Score"], f["up"] = ok.sum(axis=1), ok.iloc[:, 0]
    conf = f["up"] & (f["ext"] <= 12) & (f["risk"] <= 12) & (f["gap"] > 0) & f["fresh"]
    f["signal"] = np.select([conf & (f["vr"] >= 1.5) & f["strong"], conf], ["🟢", "🟡"], default="")
    return f, ok


def scan(x, nifty):
    f, ok = features(x, nifty)
    t = f.iloc[-1]
    age = next((a for a in range(3) if f["signal"].iloc[-1 - a]), None)   # breakout today, yesterday or 2 days ago
    s = f.iloc[-1 - age] if age is not None else t
    day = ["today", "yesterday", "2 days ago"][age] if age is not None else ""
    since = (t["Close"] / s["Close"] - 1) * 100 if age is not None else np.nan
    risk = (t["Close"] - s["stop"]) / t["Close"] * 100
    if age is not None:
        if since > 6:
            st_ = "🟠 Broke out earlier, already up - don't chase"
        elif s["signal"] == "🟢":
            st_ = "🟢 Breakout " + day
        else:
            st_ = "🟡 Breakout " + day + ", weak volume"
    elif not t["up"]:
        st_ = "🔴 Avoid (no uptrend)"
    elif t["ext"] > 12 or t["risk"] > 12:
        st_ = "🟠 Extended - don't chase"
    elif t["gap"] > 0:
        st_ = "🟠 Already running - wait for pullback"
    elif t["gap"] >= -3 and t["rng"] <= 10:
        st_ = "🔵 Near breakout - watchlist"
    elif ok["Within 10% of 52w high"].iloc[-1] and ok["Beats Nifty (3m)"].iloc[-1]:
        st_ = "⚪ Building - watchlist (far)"
    else:
        st_ = "🔴 Avoid (weak)"
    return {"Status": st_, "Breakout": day, "Days ago": age if age is not None else np.nan,
            "Score": int(t["Score"]), "Close": round(t["Close"], 2), "Breakout level": round(s["Level"], 2),
            "% vs level": round((t["Close"] / s["Level"] - 1) * 100, 1), "Since breakout %": round(since, 1),
            "Volume x": round(s["vr"], 1), "Range %": round(s["rng"], 1), "RSI": round(t["RSI"], 1),
            "Stop": round(s["stop"], 2), "Risk %": round(risk, 1), "Above 20 EMA %": round(t["ext"], 1),
            "Liquidity (Rs cr)": round((x["Close"] * x["Volume"]).iloc[-20:].mean() / 1e7, 1),
            **{k: ("✔" if b else "") for k, b in ok.iloc[-1].items()}}


def backtest(prices, nifty, horizon=20):
    """Replay every past 🟢 signal: buy at the close, stop = 10-day low, target = 2x risk, max 20 days."""
    rows = []
    for sym, x in prices.items():
        f, _ = features(x, nifty)
        h, l, c, stp = x["High"].values, x["Low"].values, x["Close"].values, f["stop"].values
        for i in np.flatnonzero((f["signal"] == "🟢").values):
            if i + horizon >= len(c) or c[i] - stp[i] <= 0:
                continue
            e, rk, r, out = c[i], c[i] - stp[i], None, "time"
            for j in range(i + 1, i + horizon + 1):
                if l[j] <= stp[i]:
                    r, out = -1.0, "stop"; break
                if h[j] >= e + 2 * rk:
                    r, out = 2.0, "target"; break
            if r is None:
                r = (c[i + horizon] - e) / rk
            rows.append({"Symbol": sym, "Date": x.index[i].date(), "R": r, "Out": out, "Ret20": c[i + horizon] / e - 1,
                         "Green": bool(f["good"].iloc[i]), "Tight": bool(f["rng"].iloc[i] <= 10)})
    return pd.DataFrame(rows)


def summarise(b):
    rows = []
    for name, m in (("All past 🟢 breakouts", b["R"].notna()), ("Only when Nifty was above 50 & 200 DMA", b["Green"]),
                    ("Only with a tight base (under 10%)", b["Tight"]), ("Both conditions", b["Green"] & b["Tight"])):
        d = b[m]
        if len(d):
            rows.append({"Setup": name, "Signals": len(d), "Hit 1:2 target %": round((d["Out"] == "target").mean() * 100),
                         "Stopped out %": round((d["Out"] == "stop").mean() * 100),
                         "Neither in 20 days %": round((d["Out"] == "time").mean() * 100),
                         "Avg R per trade": round(d["R"].mean(), 2), "Avg 20-day return %": round(d["Ret20"].mean() * 100, 1)})
    return pd.DataFrame(rows)


def guide(r):
    pos = "above" if r["% vs level"] > 0 else "below"
    pe = r.get("P/E")
    if pe is None or pd.isna(pe):
        pe_t = "P/E is not available (often loss-making or missing data), so check Screener.in."
    elif pe > 60:
        pe_t = f"P/E {pe} is high: the market expects strong growth, so moves can be sharp both ways."
    else:
        pe_t = f"P/E {pe}: compare it with 2-3 similar companies in the same industry before judging if it is cheap or costly."
    verdict = {"🟢": "Possible entry only if the market banner is green and no big resistance sits just above. Otherwise wait for a retest of the level.",
               "🟡": "Price crossed the level but volume did not confirm. Wait for a stronger close or a retest.",
               "🔵": "Add to your watchlist. Do nothing until a daily close above the breakout level on high volume.",
               "⚪": "Far from a breakout. Keep it on a long watchlist and recheck weekly.",
               "🟠": "Do not chase. Wait for a pullback to the 20 EMA or a retest of the level.",
               "🔴": "Skip it for now."}[r["Status"][0]]
    return f"""**{r['Company Name']} ({r['Symbol']}): what to check on the chart**

1. **Breakout level (orange dashed line): Rs {r['Breakout level']}.** Price is {abs(r['% vs level'])}% {pos} it. You buy only after a daily **close** above this line on strong volume.
2. **Volume (middle panel):** {'on the breakout day' if r['Breakout'] else 'today'} was {r['Volume x']}x its 50-day average. A real breakout needs about 1.5x or more. Quiet volume while price sits just below the line is a good sign.
3. **Base (last 3 weeks):** price moved inside a {r['Range %']}% range. Under 10% is tight and healthy.
4. **Trend (EMA lines):** price should sit above the 50 and 200 EMA. It is {abs(r['Above 20 EMA %'])}% {'above' if r['Above 20 EMA %'] >= 0 else 'below'} the 20 EMA; above 12% means stretched.
5. **RSI (bottom panel): {r['RSI']}.** Healthy uptrends hold between 40 and 80.
6. **Risk:** stop near Rs {r['Stop']} ({r['Risk %']}% below the price). The larger this number, the smaller your position should be.
7. **Valuation:** {pe_t}

**Verdict: {r['Status'][2:]}.** {verdict}{(' **Cautions:** ' + r['Notes'] + '.') if r.get('Notes') else ''}"""


def draw(x, sym):
    c = x["Close"]
    d = pd.DataFrame({"e20": ema(c, 20), "e50": ema(c, 50), "e200": ema(c, 200), "rsi": rsi(c),
                      "vavg": x["Volume"].rolling(50).mean()}).join(x).tail(180)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[.6, .2, .2], vertical_spacing=.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                                 name="Price", increasing_line_color="#1f8a5b", decreasing_line_color="#c2413b"), 1, 1)
    for k, name, col in (("e20", "EMA 20", "#3b82f6"), ("e50", "EMA 50", "#d98a00"), ("e200", "EMA 200", "#888888")):
        fig.add_trace(go.Scatter(x=d.index, y=d[k], name=name, line=dict(color=col, width=1.4)), 1, 1)
    fig.add_hline(y=c.iloc[-61:-1].max(), line_dash="dash", line_color="#d98a00",
                  annotation_text="Breakout level (3-month high)", row=1, col=1)
    fig.add_hline(y=x["Low"].iloc[-10:].min(), line_dash="dash", line_color="#c2413b",
                  annotation_text="Stop (10-day low)", row=1, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="Volume", marker_color="#8aa0ab"), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["vavg"], name="50-day avg volume", line=dict(color="#d98a00", width=1.2)), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["rsi"], name="RSI 14", line=dict(color="#3b82f6", width=1.4)), 3, 1)
    for lvl in (40, 80):
        fig.add_hline(y=lvl, line_dash="dot", line_color="#888888", row=3, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(height=680, title=sym, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    return fig


def compare(x, nifty, sym):
    d = pd.concat([x["Close"], nifty], axis=1, keys=[sym, "Nifty 50"]).dropna().tail(130)
    d = d / d.iloc[0] * 100
    fig = go.Figure([go.Scatter(x=d.index, y=d[k], name=k) for k in d])
    fig.update_layout(height=300, title="Last 6 months, both start at 100 (higher line = stronger)",
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    return fig


st.title("Large and mid cap scanner for swing and positional setups")
st.caption("Nifty 100 + Nifty Midcap 150 stocks. Prices: Yahoo Finance (unofficial, about 15 minutes delayed), "
           "5 years of daily data. Educational only, not financial advice. A status is a shortlist, not a prediction.")
try:
    uni = load_universe()
except Exception:
    up = st.file_uploader("Could not fetch the NSE lists. Upload any NSE index CSV (for example "
                          "ind_niftylargemidcap250list.csv from niftyindices.com)", type="csv")
    if up is None:
        st.stop()
    uni = pd.read_csv(up)[["Company Name", "Industry", "Symbol"]].assign(Cap="Large/Mid")

nifty = load_nifty()
good = nifty.iloc[-1] > nifty.rolling(50).mean().iloc[-1] and nifty.iloc[-1] > nifty.rolling(200).mean().iloc[-1]
(st.success if good else st.warning)(
    "Market check: Nifty is above its 50 and 200 DMA, so conditions favour breakouts." if good
    else "Market check: Nifty is below its 50 or 200 DMA. Trade smaller or wait.")

with st.expander("How to read the results (start here)"):
    st.markdown("""
- 🟢 **Breakout today / 1 day ago / 2 days ago:** closed above its 3-month high on strong volume, in an uptrend, and still holds above the level. Check for resistance just above and read the Notes column.
- 🟡 **Breakout (caution):** crossed the level but volume or trend is weak. Wait or study it carefully.
- 🔵 **Near breakout - watchlist:** within 3% below the level with a tight base. **Best list to study.** Set an alert at the orange line.
- ⚪ **Building (far):** strong stock but not close to a breakout. Recheck weekly.
- 🟠 **Extended / already running:** the move has happened. Do not chase; wait for a pullback.
- 🔴 **Avoid / failed breakout:** no uptrend, weak versus the Nifty, or a breakout that closed back below its level.

**Score (out of 8)** counts trend, RSI, MACD, closeness to highs, strength versus the Nifty and base tightness. Status matters more than score.
**P/E** is shown for the top 40 candidates only. **Risk %** is the gap to the suggested stop; above 8% means a smaller position.""")

min_score = st.sidebar.slider("Minimum score (out of 8)", 0, 8, 3)
min_liq = st.sidebar.number_input("Minimum daily traded value (Rs crore)", 0.0, 500.0, 5.0)
caps = st.sidebar.multiselect("Company size", ["Large", "Mid"], ["Large", "Mid"])
sectors = st.sidebar.multiselect("Industry filter (optional)", sorted(uni["Industry"].unique()))

if st.button("Run scan", type="primary"):
    prices = load_prices(tuple(uni["Symbol"]))
    st.session_state["res"] = pd.DataFrame([{"Symbol": s, **scan(x, nifty)} for s, x in prices.items()]).merge(uni, on="Symbol")
    with st.spinner("Backtesting the same rules on 5 years of history..."):
        st.session_state["bt"] = backtest(prices, nifty)

res = st.session_state.get("res")
if res is None:
    st.write("Press **Run scan** (best after 4 PM IST, when today's candle is final).")
    st.stop()

st.subheader("How accurate has this scanner been? (backtest of the same rules)")
bt = st.session_state.get("bt")
if bt is None or bt.empty:
    st.info("Not enough past 🟢 signals to test.")
else:
    sm = summarise(bt)
    a = sm.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Past signals tested", int(a["Signals"]))
    m2.metric("Hit 1:2 target first", f"{a['Hit 1:2 target %']}%")
    m3.metric("Stopped out first", f"{a['Stopped out %']}%")
    m4.metric("Average per trade", f"{a['Avg R per trade']} R")
    st.dataframe(sm, use_container_width=True, hide_index=True)
    st.caption("Method: each past day a stock met today's 🟢 rules, buy at that close, stop at the 10-day low, target at twice the "
               "risk, exit after 20 trading days if neither is hit. R = one unit of risk (-1R stop, +2R target). With a 1:2 "
               "target you break even at about a 34% hit rate. Limits: only today's Nifty 100 and Midcap 150 companies are "
               "tested (survivorship bias), costs are ignored, signals cluster in bull markets, and the past does not "
               "guarantee the future. Treat it as an estimate of reliability, not a promise.")
    st.download_button("Download backtest trades", bt.to_csv(index=False), "backtest.csv")

df = res[(res["Score"] >= min_score) & (res["Liquidity (Rs cr)"] >= min_liq)]
df = df[df["Cap"].isin(caps + ["Large/Mid"])]
if sectors:
    df = df[df["Industry"].isin(sectors)]
df = df.sort_values(["Days ago", "Score", "Volume x"], ascending=[True, False, False])
focus = df[df["Status"].str[0].isin(["🟢", "🟡", "🔵", "⚪"]) | (df["Breakout"] != "")].head(40)
with st.spinner("Fetching P/E for the top candidates..."):
    fund = {s: fundamentals(s) for s in focus["Symbol"]}
df = df.assign(**{k: df["Symbol"].map(lambda s, k=k: fund.get(s, {}).get(k)) for k in ("P/E", "Debt/Equity", "Profit growth %")})
st.subheader(f"{len(df)} stocks passed the filters, {len(res)} scanned")

COLS = ["Company Name", "Symbol", "Cap", "Status", "Breakout", "Score", "Close", "Breakout level", "% vs level",
        "Since breakout %", "Volume x", "Range %", "RSI", "Stop", "Risk %", "P/E", "Industry"]
S0 = df["Status"].str[0]
groups = [("🟢 Breakouts (last 3 days)", df["Breakout"] != ""), ("🔵 Watchlist", S0.isin(["🔵", "⚪"]) & (df["Breakout"] == "")),
          ("🟠 Don't chase", (S0 == "🟠") & (df["Breakout"] == "")), ("🔴 Avoid", S0 == "🔴"), ("All", S0.notna())]
for tab, (name, mask) in zip(st.tabs([g[0] for g in groups]), groups):
    with tab:
        d = df[mask]
        if name.startswith("🟢") and d.empty:
            st.info("No breakouts in the last 3 days among the filtered stocks. Try lowering the minimum score or liquidity. "
                    "Quiet or weak markets can have very few real breakouts, and that is normal.")
        st.dataframe(d[COLS], use_container_width=True, hide_index=True)
        st.caption(f"{len(d)} stocks. 'Range %' is the width of the base before the breakout (under 10% is tight).")
with st.expander("Show every check for every stock"):
    st.dataframe(df, use_container_width=True, hide_index=True)
st.download_button("Download CSV", df.to_csv(index=False), "scan.csv")

if len(df):
    order = list(focus["Symbol"]) + [s for s in df["Symbol"] if s not in set(focus["Symbol"])]
    names = dict(zip(df["Symbol"], df["Company Name"]))
    sym = st.selectbox("Study a stock", order, format_func=lambda s: f"{names[s]} ({s})")
    row = df[df["Symbol"] == sym].iloc[0]
    px = load_prices(tuple(uni["Symbol"])).get(sym)
    st.markdown(guide(row))
    if px is not None:
        st.plotly_chart(draw(px, sym), use_container_width=True)
        st.plotly_chart(compare(px, nifty, sym), use_container_width=True)
    passed = [k for k in row.index if row[k] == "✔"]
    failed = [k for k in row.index if row[k] == ""]
    st.success("Checks passed: " + (", ".join(passed) or "none"))
    st.warning("Checks not passed: " + (", ".join(failed) or "none"))
    st.link_button("Also open on TradingView (optional)", "https://in.tradingview.com/chart/?symbol=NSE%3A" + quote(sym, safe=""))
