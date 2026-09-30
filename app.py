"""Large + mid cap NSE scanner for swing and positional setups. Educational only, not advice."""
import io
from urllib.parse import quote

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


@st.cache_data(ttl=1800, show_spinner="Downloading 2 years of prices (first run takes a minute or two)...")
def load_prices(symbols: tuple):
    out = {}
    for i in range(0, len(symbols), 100):
        tick = [s + ".NS" for s in symbols[i:i + 100]]
        d = yf.download(tick, period="2y", interval="1d", group_by="ticker",
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
    d = yf.download("^NSEI", period="2y", interval="1d", auto_adjust=True, progress=False)
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


def analyse(x, nifty):
    c, v, h, l = x["Close"], x["Volume"], x["High"], x["Low"]
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    macd = ema(c, 12) - ema(c, 26)
    sig = ema(macd, 9)
    last, r = c.iloc[-1], rsi(c).iloc[-1]
    level = c.iloc[-61:-1].max()                       # highest close of the last ~3 months
    gap = (last / level - 1) * 100                     # + above the level, - below it
    fresh = c.iloc[-2] <= c.iloc[-62:-2].max()         # was yesterday still below the level?
    vr = v.iloc[-1] / v.iloc[-51:-1].mean()
    rng = (h.iloc[-16:-1].max() - l.iloc[-16:-1].min()) / last * 100
    stop = l.iloc[-10:].min()
    risk = (last - stop) / last * 100
    ext = (last / e20.iloc[-1] - 1) * 100
    strong = last >= l.iloc[-1] + 0.75 * (h.iloc[-1] - l.iloc[-1])
    n3 = nifty.iloc[-1] / nifty.iloc[-64] - 1
    n6 = nifty.iloc[-1] / nifty.iloc[-127] - 1
    ok = {
        "Uptrend (price>50>200 EMA, 200 rising)": bool(last > e50.iloc[-1] > e200.iloc[-1] and e200.iloc[-1] > e200.iloc[-21]),
        "RSI 50-80": bool(50 <= r <= 80),
        "MACD up and above 0": bool(macd.iloc[-1] > sig.iloc[-1] and macd.iloc[-1] > 0),
        "At or within 3% of 3-month high": bool(gap >= -3),
        "Within 10% of 52w high": bool(last >= 0.9 * c.iloc[-252:].max()),
        "Beats Nifty (3m)": bool(last / c.iloc[-64] - 1 > n3),
        "Beats Nifty (6m)": bool(last / c.iloc[-127] - 1 > n6),
        "Tight 3-week base (range under 10%)": bool(rng <= 10),
    }
    if not list(ok.values())[0]:
        s = "🔴 Avoid (no uptrend)"
    elif ext > 12 or risk > 12:
        s = "🟠 Extended - don't chase"
    elif gap > 0:
        s = ("🟢 Fresh breakout today" if vr >= 1.5 and strong else "🟡 Breakout, weak volume") if fresh \
            else "🟠 Already running - wait for pullback"
    elif gap >= -3 and rng <= 10:
        s = "🔵 Near breakout - watchlist"
    elif ok["Within 10% of 52w high"] and ok["Beats Nifty (3m)"]:
        s = "⚪ Building - watchlist (far)"
    else:
        s = "🔴 Avoid (weak)"
    return {"Status": s, "Score": int(sum(ok.values())), "Close": round(last, 2), "Breakout level": round(level, 2),
            "% vs level": round(gap, 1), "Volume x": round(vr, 1), "RSI": round(r, 1), "Range %": round(rng, 1),
            "Stop": round(stop, 2), "Risk %": round(risk, 1), "Above 20 EMA %": round(ext, 1),
            "Liquidity (Rs cr)": round((c * v).iloc[-20:].mean() / 1e7, 1),
            **{k: ("✔" if b else "") for k, b in ok.items()}}


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
2. **Volume (middle panel):** today is {r['Volume x']}x its 50-day average. A real breakout needs about 1.5x or more. Quiet volume while price sits just below the line is a good sign.
3. **Base (last 3 weeks):** price moved inside a {r['Range %']}% range. Under 10% is tight and healthy.
4. **Trend (EMA lines):** price should sit above the 50 and 200 EMA. It is {abs(r['Above 20 EMA %'])}% {'above' if r['Above 20 EMA %'] >= 0 else 'below'} the 20 EMA; above 12% means stretched.
5. **RSI (bottom panel): {r['RSI']}.** Healthy uptrends hold between 40 and 80.
6. **Risk:** stop near Rs {r['Stop']} ({r['Risk %']}% below the price). The larger this number, the smaller your position should be.
7. **Valuation:** {pe_t}

**Verdict: {r['Status'][2:]}.** {verdict}"""


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
           "2 years of daily data. Educational only, not financial advice. A status is a shortlist, not a prediction.")
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
- 🟢 **Fresh breakout today:** crossed the 3-month high today on strong volume. Check the chart for resistance just above, then consider an entry plan.
- 🟡 **Breakout, weak volume:** crossed the level but buyers are not convincing. Wait.
- 🔵 **Near breakout - watchlist:** within 3% below the level with a tight base. **Best list to study.** Set an alert at the orange line.
- ⚪ **Building (far):** strong stock but not close to a breakout. Recheck weekly.
- 🟠 **Extended / already running:** the move has happened. Do not chase; wait for a pullback.
- 🔴 **Avoid:** no uptrend or weak versus the Nifty.

**Score (out of 8)** counts trend, RSI, MACD, closeness to highs, strength versus the Nifty and base tightness. Status matters more than score.
**P/E** is shown for the top 40 candidates only. **Risk %** is the gap to the suggested stop; above 8% means a smaller position.""")

min_score = st.sidebar.slider("Minimum score (out of 8)", 0, 8, 5)
min_liq = st.sidebar.number_input("Minimum daily traded value (Rs crore)", 0.0, 500.0, 10.0)
caps = st.sidebar.multiselect("Company size", ["Large", "Mid"], ["Large", "Mid"])
sectors = st.sidebar.multiselect("Industry filter (optional)", sorted(uni["Industry"].unique()))

if st.button("Run scan", type="primary"):
    prices = load_prices(tuple(uni["Symbol"]))
    rows = [{"Symbol": s, **analyse(x, nifty)} for s, x in prices.items()]
    st.session_state["res"] = pd.DataFrame(rows).merge(uni, on="Symbol")

res = st.session_state.get("res")
if res is None:
    st.write("Press **Run scan** (best after 4 PM IST, when today's candle is final).")
    st.stop()

df = res[(res["Score"] >= min_score) & (res["Liquidity (Rs cr)"] >= min_liq)]
df = df[df["Cap"].isin(caps + ["Large/Mid"])]
if sectors:
    df = df[df["Industry"].isin(sectors)]
df = df.sort_values(["Score", "Volume x"], ascending=False)
focus = df[df["Status"].str[0].isin(["🟢", "🟡", "🔵", "⚪"])].head(40)
with st.spinner("Fetching P/E for the top candidates..."):
    fund = {s: fundamentals(s) for s in focus["Symbol"]}
df = df.assign(**{k: df["Symbol"].map(lambda s, k=k: fund.get(s, {}).get(k)) for k in ("P/E", "Debt/Equity", "Profit growth %")})
st.subheader(f"{len(df)} stocks passed the filters, {len(res)} scanned")

COLS = ["Company Name", "Symbol", "Cap", "Status", "Score", "Close", "Breakout level", "% vs level",
        "Volume x", "RSI", "Stop", "Risk %", "P/E", "Industry"]
groups = [("🟢 Breakout", ("🟢", "🟡")), ("🔵 Watchlist", ("🔵", "⚪")), ("🟠 Don't chase", ("🟠",)),
          ("🔴 Avoid", ("🔴",)), ("All", ("🟢", "🟡", "🔵", "⚪", "🟠", "🔴"))]
for tab, (name, icons) in zip(st.tabs([g[0] for g in groups]), groups):
    with tab:
        d = df[df["Status"].str[0].isin(icons)]
        st.dataframe(d[COLS], use_container_width=True, hide_index=True)
        st.caption(f"{len(d)} stocks")
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
