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


def adx_atr(h, l, c, n=14):
    """Wilder ADX (trend strength) and ATR (typical daily range), idea borrowed from the uploaded repo."""
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), atr


def features(x, nifty):
    """One row per day, using exactly the same rules for today's scan and for the backtest."""
    c, v, h, l = x["Close"], x["Volume"], x["High"], x["Low"]
    n = nifty.reindex(x.index).ffill()
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    macd = ema(c, 12) - ema(c, 26)
    adx14, atr = adx_atr(h, l, c)
    f = pd.DataFrame(index=x.index)
    f["Close"], f["RSI"], f["e20"] = c, rsi(c), e20
    f["Level"] = c.shift(1).rolling(60).max()                  # highest close of the prior ~3 months
    f["gap"] = (c / f["Level"] - 1) * 100
    f["fresh"] = c.shift(1) <= f["Level"].shift(1)             # yesterday was still below the level
    f["vr"] = v / v.shift(1).rolling(50).mean()
    f["rng"] = (h.shift(1).rolling(15).max() - l.shift(1).rolling(15).min()) / c * 100
    f["stop"] = l.rolling(10).min()
    f["risk"] = (c - f["stop"]) / c * 100
    f["ext"] = (c / e20 - 1) * 100
    f["strong"] = (c >= l + 0.75 * (h - l)) & ((h - l) <= 2.5 * atr)   # closes near the high, candle not oversized
    f["nr7"] = (h - l) <= (h - l).rolling(7).min()                      # narrowest range of the last 7 days
    f["good"] = (n > n.rolling(50).mean()) & (n > n.rolling(200).mean())
    ok = pd.DataFrame({
        "Uptrend (price>50>200 EMA, 200 rising)": (c > e50) & (e50 > e200) & (e200 > e200.shift(20)),
        "RSI 50-80": f["RSI"].between(50, 80),
        "MACD up and above 0": (macd > ema(macd, 9)) & (macd > 0),
        "Trend strength (ADX 20+)": adx14 >= 20,
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
            "Volume x": round(s["vr"], 1), "Range %": round(s["rng"], 1), "RSI": round(t["RSI"], 1), "6m return %": round((t["Close"] / x["Close"].iloc[-127] - 1) * 100, 1),
            "3m return %": round((t["Close"] / x["Close"].iloc[-64] - 1) * 100, 1), "NR7": "✔" if t["nr7"] else "",
            "Stop": round(s["stop"], 2), "Risk %": round(risk, 1), "Above 20 EMA %": round(t["ext"], 1),
            "Liquidity (Rs cr)": round((x["Close"] * x["Volume"]).iloc[-20:].mean() / 1e7, 1),
            **{k: ("✔" if b else "") for k, b in ok.iloc[-1].items()}}


def backtest(prices, nifty, horizon=20, trail_days=60):
    """Replay every past 🟢 signal like a real trader: buy NEXT day's open, stop = signal-day 10-day low,
    gap-downs fill at the open (worse than the stop), two exits compared: fixed 1:2 target vs 20 EMA trailing exit."""
    rows = []
    for sym, x in prices.items():
        f, _ = features(x, nifty)
        o, h, l, c = (x[k].values for k in ("Open", "High", "Low", "Close"))
        stp, e20 = f["stop"].values, f["e20"].values
        for i in np.flatnonzero((f["signal"] == "🟢").values):
            if i + trail_days + 1 >= len(c):
                continue
            e = o[i + 1]
            rk = e - stp[i]
            if rk <= 0 or rk / e > 0.15:
                continue
            r, out = None, "time"
            for j in range(i + 1, i + horizon + 1):
                if l[j] <= stp[i]:
                    r, out = (min(stp[i], o[j]) - e) / rk, "stop"; break
                if h[j] >= e + 2 * rk:
                    r, out = (max(e + 2 * rk, o[j]) - e) / rk, "target"; break
            if r is None:
                r = (c[i + horizon] - e) / rk
            rt = None
            for j in range(i + 1, i + trail_days + 1):
                if l[j] <= stp[i]:
                    rt = (min(stp[i], o[j]) - e) / rk; break
                if c[j] < e20[j]:
                    rt = (c[j] - e) / rk; break
            if rt is None:
                rt = (c[i + trail_days] - e) / rk
            rows.append({"Symbol": sym, "Date": x.index[i].date(), "Year": x.index[i].year, "R": r, "Out": out, "R2": rt,
                         "Ret20": c[i + horizon] / e - 1, "CostR": e / rk, "Green": bool(f["good"].iloc[i]),
                         "Tight": bool(f["rng"].iloc[i] <= 10)})
    return pd.DataFrame(rows)


def _stats(d, cost):
    rn, r2 = d["R"] - cost / 100 * d["CostR"], d["R2"] - cost / 100 * d["CostR"]
    neg = -rn[rn < 0].sum()
    return {"Signals": len(d), "Hit 1:2 target %": round((d["Out"] == "target").mean() * 100),
            "Stopped out %": round((d["Out"] == "stop").mean() * 100),
            "Avg R (fixed 1:2 exit)": round(rn.mean(), 2), "Avg R (trail 20 EMA exit)": round(r2.mean(), 2),
            "Profit factor": round(rn[rn > 0].sum() / neg, 2) if neg > 0 else None}


def summarise(b, cost):
    rows = []
    for name, m in (("All past 🟢 breakouts", b["R"].notna()), ("Only when Nifty was above 50 & 200 DMA", b["Green"]),
                    ("Only with a tight base (under 10%)", b["Tight"]), ("Both conditions", b["Green"] & b["Tight"])):
        if m.any():
            rows.append({"Setup": name, **_stats(b[m], cost)})
    return pd.DataFrame(rows)


def by_year(b, cost):
    return pd.DataFrame([{"Year": y, **_stats(d, cost)} for y, d in b.groupby("Year")])


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

**Score (out of 9)** counts trend, trend strength (ADX), RSI, MACD, closeness to highs, strength versus the Nifty and base tightness. Status matters more than score.
**Sector rank** orders industries by their median 3-month return (1 = strongest). **RS rank %** ranks the stock's 6-month return among all scanned stocks (80+ = market leader). **P/E** is shown for the top 40 candidates only. **Risk %** is the gap to the suggested stop; above 8% means a smaller position.""")

min_score = st.sidebar.slider("Minimum score (out of 9)", 0, 9, 3)
min_liq = st.sidebar.number_input("Minimum daily traded value (Rs crore)", 0.0, 500.0, 5.0)
caps = st.sidebar.multiselect("Company size", ["Large", "Mid"], ["Large", "Mid"])
sectors = st.sidebar.multiselect("Industry filter (optional)", sorted(uni["Industry"].unique()))
top_sec = st.sidebar.slider("Only show the top N strongest sectors (0 = all)", 0, 30, 0)
st.sidebar.markdown("**Your trade plan**")
capital = st.sidebar.number_input("Trading capital (Rs)", 10000, 100000000, 200000, step=10000)
risk_pct = st.sidebar.number_input("Risk per trade (% of capital)", 0.1, 5.0, 1.0, step=0.1)
cost_pct = st.sidebar.number_input("Round-trip cost + slippage assumed in backtest (%)", 0.0, 2.0, 0.3, step=0.05)

if st.button("Run scan", type="primary"):
    prices = load_prices(tuple(uni["Symbol"]))
    r0 = pd.DataFrame([{"Symbol": s, **scan(x, nifty)} for s, x in prices.items()]).merge(uni, on="Symbol")
    r0["RS rank %"] = (r0["6m return %"].rank(pct=True) * 100).round()
    sec = r0.groupby("Industry")["3m return %"].median()
    r0["Sector 3m %"] = r0["Industry"].map(sec).round(1)
    r0["Sector rank"] = r0["Industry"].map(sec.rank(ascending=False)).round().astype("Int64")
    st.session_state["res"] = r0
    with st.spinner("Backtesting the same rules on 5 years of history..."):
        st.session_state["bt"] = backtest(prices, nifty)

res = st.session_state.get("res")
if res is None:
    st.write("Press **Run scan** (best after 4 PM IST, when today's candle is final).")
    st.stop()

st.subheader("How accurate has this scanner been? (realistic backtest of the same rules)")
bt = st.session_state.get("bt")
if bt is None or bt.empty:
    st.info("Not enough past 🟢 signals to test.")
else:
    sm = summarise(bt, cost_pct)
    a = sm.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Past signals tested", int(a["Signals"]))
    m2.metric("Hit 1:2 target first", f"{a['Hit 1:2 target %']}%")
    m3.metric("Stopped out first", f"{a['Stopped out %']}%")
    m4.metric("Avg result per trade (after costs)", f"{a['Avg R (fixed 1:2 exit)']} R")
    if a["Avg R (fixed 1:2 exit)"] <= 0:
        st.error("After costs the fixed 1:2 exit did not make money on past data. Treat this scanner as a study tool only and paper trade.")
    else:
        st.success("Positive average result on past data, but it is an estimate, not a promise. Paper trade before using real money.")
    st.dataframe(sm, use_container_width=True, hide_index=True)
    st.markdown("**Is it consistent? Results by year** (big swings between years mean the edge is unreliable)")
    st.dataframe(by_year(bt, cost_pct), use_container_width=True, hide_index=True)
    st.caption("Method: each past day a stock met today's 🟢 rules, buy at the NEXT day's open (as a real trader would), stop at the "
               "signal day's 10-day low, gaps below the stop fill at the open, costs deducted. Two exits are compared: a fixed 1:2 target "
               "(20 days) and riding the trend until a close below the 20 EMA (60 days). R = one unit of risk. A 1:2 target breaks "
               "even near a 34% hit rate. Limits: only today's Nifty 100 and Midcap 150 companies are tested (survivorship bias makes "
               "results look better than real life), signals cluster in bull markets, and results/news surprises are not modelled.")
    st.download_button("Download backtest trades", bt.to_csv(index=False), "backtest.csv")

df = res[(res["Score"] >= min_score) & (res["Liquidity (Rs cr)"] >= min_liq)]
df = df[df["Cap"].isin(caps + ["Large/Mid"])]
if sectors:
    df = df[df["Industry"].isin(sectors)]
if top_sec:
    df = df[(df["Sector rank"] <= top_sec).fillna(False)]
df = df.sort_values(["Days ago", "Score", "Volume x"], ascending=[True, False, False])
focus = df[df["Status"].str[0].isin(["🟢", "🟡", "🔵", "⚪"]) | (df["Breakout"] != "")].head(40)
with st.spinner("Fetching P/E for the top candidates..."):
    fund = {s: fundamentals(s) for s in focus["Symbol"]}
df = df.assign(**{k: df["Symbol"].map(lambda s, k=k: fund.get(s, {}).get(k)) for k in ("P/E", "Debt/Equity", "Profit growth %")})
st.subheader(f"{len(df)} stocks passed the filters, {len(res)} scanned")

COLS = ["Company Name", "Symbol", "Cap", "Status", "Breakout", "Score", "Close", "Breakout level", "% vs level",
        "Since breakout %", "Volume x", "RS rank %", "Range %", "RSI", "Stop", "Risk %", "P/E", "Sector rank", "Industry"]
S0 = df["Status"].str[0]
groups = [("🟢 Breakouts (last 3 days)", df["Breakout"] != ""), ("🔵 Watchlist", S0.isin(["🔵", "⚪"]) & (df["Breakout"] == "")),
          ("🟠 Don't chase", (S0 == "🟠") & (df["Breakout"] == "")), ("🔴 Avoid", S0 == "🔴"), ("All", S0.notna())]
for tab, (name, mask) in zip(st.tabs([g[0] for g in groups]), groups):
    with tab:
        d = df[mask]
        if name.startswith("🟢") and d.empty:
            st.info("No breakouts in the last 3 days among the filtered stocks. Try lowering the minimum score or liquidity. "
                    "Quiet or weak markets can have very few real breakouts, and that is normal.")
        if name.startswith("🟢") and not good:
            st.warning("Nifty is below its 50 or 200 DMA. Compare the backtest rows above for this condition and be extra cautious or skip.")
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
    per = row["Close"] - row["Stop"]
    if per > 0:
        qty = int(capital * risk_pct / 100 / per)
        st.info(f"**Position size for your plan:** risk Rs {capital * risk_pct / 100:,.0f} / Rs {per:.2f} per share = **{qty} shares** "
                f"(about Rs {qty * row['Close']:,.0f}, {qty * row['Close'] / capital * 100:.0f}% of capital). 1:2 target near Rs {row['Close'] + 2 * per:.2f}. "
                f"Fifteen losses in a row at this risk would cost about {(1 - (1 - risk_pct / 100) ** 15) * 100:.0f}% of capital, so keep risk small. "
                "Before entering, check the company's results date and recent news; scanners cannot see event risk.")
        if qty * row["Close"] > 0.20 * capital:
            st.warning("This position is over 20% of your capital. Consider fewer shares or skip it.")
    if px is not None:
        st.plotly_chart(draw(px, sym), use_container_width=True)
        st.plotly_chart(compare(px, nifty, sym), use_container_width=True)
    chk = list(features(px, nifty)[1].columns) if px is not None else []
    passed = [k for k in chk if row[k] == "✔"]
    failed = [k for k in chk if row[k] == ""]
    st.success("Checks passed: " + (", ".join(passed) or "none"))
    st.warning("Checks not passed: " + (", ".join(failed) or "none"))
    st.link_button("Also open on TradingView (optional)", "https://in.tradingview.com/chart/?symbol=NSE%3A" + quote(sym, safe=""))
