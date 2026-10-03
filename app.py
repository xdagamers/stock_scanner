"""Large + mid cap NSE scanner for swing and positional setups. Educational only, not advice."""
import io
from datetime import datetime
from html import escape
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="NSE Swing Scanner", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
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


COLS = ["Company Name", "Symbol", "Cap", "Status", "Breakout", "Score", "Close", "Breakout level", "% vs level",
        "Since breakout %", "Volume x", "RS rank %", "Range %", "RSI", "Stop", "Risk %", "P/E", "Sector rank", "Industry"]


CSS = """<style>
.block-container{padding-top:1rem;max-width:1100px}
.hero{display:flex;gap:12px;flex-wrap:wrap;justify-content:space-between;align-items:center;padding:18px 20px;border-radius:16px;background:linear-gradient(120deg,#4f46e5,#0891b2);color:#fff;margin-bottom:10px}
.hero h1{margin:0;padding:0;font-size:1.6rem;color:#fff}.hero p{margin:2px 0 0;opacity:.92;color:#fff}
.mk{padding:8px 14px;border-radius:12px;color:#fff;font-weight:700;display:flex;flex-direction:column}.mk small{font-weight:400}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:8px 0 14px}
.kpi{background:rgba(127,127,127,.10);border-radius:12px;padding:12px 14px}
.kv{font-size:1.7rem;font-weight:800;line-height:1.1}.kl{font-weight:600;font-size:.85rem}.ks{font-size:.72rem;opacity:.7}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.75rem;font-weight:700;white-space:normal;text-align:center}
.scard{background:rgba(127,127,127,.09);border-radius:12px;padding:12px 14px;margin:8px 0}
.scard .top{display:flex;justify-content:space-between;gap:8px;align-items:flex-start}
.scard small{display:block;opacity:.72}
.scard .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px}
.scard .grid div{display:flex;flex-direction:column;font-size:.95rem}.scard .grid span{font-size:.68rem;opacity:.65}
.chip{display:inline-block;margin:3px;padding:3px 10px;border-radius:8px;font-size:.78rem}
div[data-baseweb="tab-list"]{position:sticky;top:3.5rem;z-index:50;gap:4px;overflow-x:auto;backdrop-filter:blur(10px);background:rgba(127,127,127,.14);border-radius:12px;padding:2px 6px}
@media(max-width:640px){.block-container{padding:.5rem .6rem 3rem}.hero h1{font-size:1.2rem}.scard .grid{grid-template-columns:repeat(2,1fr)}.kv{font-size:1.4rem}}
</style>"""
COL = {"🟢": "#16a34a", "🟡": "#ca8a04", "🔵": "#2563eb", "⚪": "#64748b", "🟠": "#ea580c", "🔴": "#dc2626"}


def pill(text, color):
    return f'<span class="pill" style="background:{color}26;color:{color};border:1px solid {color}66">{escape(str(text))}</span>'


def kpi(label, value, color, sub=""):
    return (f'<div class="kpi" style="border-top:4px solid {color}"><div class="kv" style="color:{color}">{value}</div>'
            f'<div class="kl">{label}</div><div class="ks">{sub}</div></div>')


def fmt(v, suf="", nd=1):
    return "–" if v is None or pd.isna(v) else f"{v:,.{nd}f}{suf}"


def card(r):
    c = COL[r["Status"][0]]
    m = [("Price", f"₹{r['Close']:,.2f}"), ("Level", f"₹{r['Breakout level']:,.2f}"), ("vs level", fmt(r["% vs level"], "%")),
         ("Volume", fmt(r["Volume x"], "x")), ("RS rank", fmt(r["RS rank %"], nd=0)), ("Risk", fmt(r["Risk %"], "%")),
         ("P/E", fmt(r.get("P/E"))), ("Score", f"{int(r['Score'])}/9")]
    cells = "".join(f"<div><b>{v}</b><span>{k}</span></div>" for k, v in m)
    return (f'<div class="scard" style="border-left:6px solid {c}"><div class="top"><div><b>{escape(str(r["Company Name"]))}</b>'
            f'<small>{escape(str(r["Symbol"]))} · {r["Cap"]} · {escape(str(r["Industry"]))}</small></div>{pill(r["Status"][2:], c)}</div>'
            f'<div class="grid">{cells}</div></div>')


def tint(v):
    c = COL.get(str(v)[:1])
    return f"background-color:{c}33;font-weight:600" if c else ""


def styled(d):
    sty = d.style
    return (sty.map if hasattr(sty, "map") else sty.applymap)(tint, subset=[k for k in ("Status",) if k in d.columns])


def guide_view():
    items = [("🟢", "Breakout (today, 1 or 2 days ago)", "Closed above its 3-month high on strong volume in an uptrend. Check for resistance just above and prefer a green market."),
             ("🟡", "Breakout, caution", "Crossed the level but volume or trend is weak. Study it, do not rush."),
             ("🔵", "Near breakout - watchlist", "Within 3% below the level with a tight base. Best list to study. Set an alert at the level."),
             ("⚪", "Building (far)", "Strong stock, not close to a breakout yet. Recheck weekly."),
             ("🟠", "Extended / already running", "The move has happened. Do not chase; wait for a pullback."),
             ("🔴", "Avoid", "No uptrend, weak versus the Nifty, or a failed breakout.")]
    st.markdown("".join(f'<div class="scard" style="border-left:6px solid {COL[i]}"><b>{i} {t}</b><small>{d}</small></div>' for i, t, d in items),
                unsafe_allow_html=True)
    with st.expander("Columns and terms"):
        st.markdown("**Score (out of 9)** counts trend, trend strength (ADX), RSI, MACD, closeness to highs, strength versus the Nifty and base tightness. "
                    "Status matters more than score. **Level** is the highest close of the last 3 months. **vs level** is the distance from it. "
                    "**Volume** compares with the 50-day average (1.5x+ confirms a breakout). **RS rank** ranks the 6-month return among all scanned "
                    "stocks (80+ = leader). **Risk** is the gap to the suggested stop (above 8% means a smaller position). **Sector rank** 1 = strongest "
                    "industry. **P/E** is shown for the top 40 candidates only. Educational only, not financial advice.")


st.markdown(CSS, unsafe_allow_html=True)
try:
    uni = load_universe()
except Exception:
    up = st.file_uploader("Could not fetch the NSE lists. Upload any NSE index CSV (for example "
                          "ind_niftylargemidcap250list.csv from niftyindices.com)", type="csv")
    if up is None:
        st.stop()
    uni = pd.read_csv(up)[["Company Name", "Industry", "Symbol"]].assign(Cap="Large/Mid")

nifty = load_nifty()
n_last, d50, d200 = nifty.iloc[-1], nifty.rolling(50).mean().iloc[-1], nifty.rolling(200).mean().iloc[-1]
good = n_last > d50 and n_last > d200
mk = ("#16a34a", "MARKET GREEN") if good else ("#ea580c", "MARKET CAUTION")
st.markdown(f'<div class="hero"><div><h1>📈 NSE Swing Scanner</h1><p>Large + mid cap breakouts, watchlist and an honest backtest</p></div>'
            f'<div class="mk" style="background:{mk[0]}">{mk[1]}<small>Nifty {n_last:,.0f} · {(n_last / d50 - 1) * 100:+.1f}% vs 50 DMA · '
            f'{(n_last / d200 - 1) * 100:+.1f}% vs 200 DMA</small></div></div>', unsafe_allow_html=True)

b1, b2 = st.columns([1, 2])
run = b1.button("🔄 Run scan", type="primary", use_container_width=True)
b2.caption(f"Last scan: {st.session_state['ts']}. Yahoo data, about 15 min delayed." if "ts" in st.session_state
           else "Best after 4 PM IST when the day's candle is final. The first run takes 1-3 minutes.")

with st.expander("⚙️ Filters and trade plan"):
    f1, f2 = st.columns(2)
    min_score = f1.slider("Minimum score (out of 9)", 0, 9, 3)
    top_sec = f2.slider("Top N strongest sectors (0 = all)", 0, 30, 0)
    min_liq = f1.number_input("Min daily traded value (Rs crore)", 0.0, 500.0, 5.0)
    caps = f2.multiselect("Company size", ["Large", "Mid"], ["Large", "Mid"])
    sectors = st.multiselect("Industry filter (optional)", sorted(uni["Industry"].unique()))
    capital = f1.number_input("Trading capital (Rs)", 10000, 100000000, 200000, step=10000)
    risk_pct = f2.number_input("Risk per trade (%)", 0.1, 5.0, 1.0, step=0.1)
    cost_pct = f1.number_input("Backtest costs + slippage (%)", 0.0, 2.0, 0.3, step=0.05)

if run:
    prices = load_prices(tuple(uni["Symbol"]))
    r0 = pd.DataFrame([{"Symbol": s, **scan(x, nifty)} for s, x in prices.items()]).merge(uni, on="Symbol")
    r0["RS rank %"] = (r0["6m return %"].rank(pct=True) * 100).round()
    sec = r0.groupby("Industry")["3m return %"].median()
    r0["Sector 3m %"] = r0["Industry"].map(sec).round(1)
    r0["Sector rank"] = r0["Industry"].map(sec.rank(ascending=False)).round().astype("Int64")
    st.session_state["res"] = r0
    st.session_state["ts"] = datetime.now().strftime("%d %b, %H:%M")
    with st.spinner("Backtesting the same rules on 5 years of history..."):
        st.session_state["bt"] = backtest(prices, nifty)
    st.rerun()

res = st.session_state.get("res")
if res is None:
    st.markdown('<div class="scard" style="border-left:6px solid #4f46e5"><b>👋 Start here</b><small>1) Tap <b>Run scan</b> above. '
                '2) Open <b>Home</b> for the summary, <b>Scanner</b> for the lists, <b>Study</b> for one stock. '
                '3) Check <b>Accuracy</b> before trusting any signal.</small></div>', unsafe_allow_html=True)
    guide_view()
    st.stop()

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
S0 = df["Status"].str[0]
buckets = {"🟢 Breakouts": df["Breakout"] != "", "🔵 Watchlist": S0.isin(["🔵", "⚪"]) & (df["Breakout"] == ""),
           "🟠 Don't chase": (S0 == "🟠") & (df["Breakout"] == ""), "🔴 Avoid": S0 == "🔴", "All": S0.notna()}
bt = st.session_state.get("bt")
sm = summarise(bt, cost_pct) if bt is not None and not bt.empty else None

t_home, t_scan, t_study, t_acc, t_guide = st.tabs(["🏠 Home", "🔍 Scanner", "📈 Study", "🎯 Accuracy", "📘 Guide"])

with t_home:
    cnt = {k: int(m.sum()) for k, m in buckets.items()}
    cards = [kpi("Breakouts (3 days)", cnt["🟢 Breakouts"], COL["🟢"], "act on with a plan"), kpi("Watchlist", cnt["🔵 Watchlist"], COL["🔵"], "wait for the level"),
             kpi("Don't chase", cnt["🟠 Don't chase"], COL["🟠"], "move already happened"), kpi("Avoid", cnt["🔴 Avoid"], COL["🔴"], "weak or no uptrend")]
    if sm is not None:
        a = sm.iloc[0]
        cards += [kpi("Past hit rate", f"{a['Hit 1:2 target %']}%", "#7c3aed", "reached 1:2 target first"),
                  kpi("Avg per trade", f"{a['Avg R (fixed 1:2 exit)']} R", COL["🟢"] if a["Avg R (fixed 1:2 exit)"] > 0 else COL["🔴"], "after costs")]
    st.markdown('<div class="kpis">' + "".join(cards) + "</div>", unsafe_allow_html=True)
    for title, key in (("🟢 Latest breakouts", "🟢 Breakouts"), ("🔵 Top watchlist", "🔵 Watchlist")):
        d = df[buckets[key]].head(5)
        st.markdown(f"#### {title}")
        if d.empty:
            st.info("None right now. This is normal in quiet or weak markets. Try lowering the minimum score in the filters." if key.startswith("🟢") else "None match your filters.")
        else:
            st.markdown("".join(card(r) for _, r in d.iterrows()), unsafe_allow_html=True)
    secs = res.groupby("Industry")["3m return %"].median().sort_values(ascending=False)
    top = pd.concat([secs.head(6), secs.tail(3)]).drop_duplicates()
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=[COL["🟢"] if v >= 0 else COL["🔴"] for v in top.values]))
    fig.update_layout(height=340, title="Sector strength: median 3-month return % (strongest and weakest)", yaxis=dict(autorange="reversed"),
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with t_scan:
    pick = st.radio("List", [f"{k} ({int(m.sum())})" for k, m in buckets.items()], horizontal=True, label_visibility="collapsed")
    name = list(buckets)[[f"{k} ({int(m.sum())})" for k, m in buckets.items()].index(pick)]
    d = df[buckets[name]]
    if name.startswith("🟢") and not good:
        st.warning("Nifty is below its 50 or 200 DMA. Check the Accuracy tab for how breakouts did in this condition, and be extra cautious.")
    if d.empty:
        st.info("No stocks in this list with your filters.")
    elif st.toggle("Table view (all rows)", False):
        more = st.toggle("Show all columns", False)
        cols = COLS if more else ["Company Name", "Status", "Breakout", "Score", "Close", "% vs level", "Volume x", "RS rank %", "Risk %", "P/E"]
        st.dataframe(styled(d[cols]), use_container_width=True, hide_index=True, height=520, column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=9, format="%d"),
            "RS rank %": st.column_config.ProgressColumn("RS rank", min_value=0, max_value=100, format="%d")})
    else:
        st.markdown("".join(card(r) for _, r in d.head(30).iterrows()), unsafe_allow_html=True)
        if len(d) > 30:
            st.caption(f"Showing the top 30 of {len(d)}. Turn on Table view to see all.")
    st.download_button("⬇️ Download this scan (CSV)", df.to_csv(index=False), "scan.csv", use_container_width=True)
    with st.expander("Every check for every stock"):
        st.dataframe(df, use_container_width=True, hide_index=True)

with t_study:
    if df.empty:
        st.info("No stocks match your filters. Loosen them in the Filters section at the top.")
    else:
        order = list(focus["Symbol"]) + [s for s in df["Symbol"] if s not in set(focus["Symbol"])]
        names = dict(zip(df["Symbol"], df["Company Name"]))
        sym = st.selectbox("Choose a stock (best candidates first)", order, format_func=lambda s: f"{names[s]} ({s})")
        row = df[df["Symbol"] == sym].iloc[0]
        px = load_prices(tuple(uni["Symbol"])).get(sym)
        st.markdown(card(row), unsafe_allow_html=True)
        k1, k2, k3, k4 = st.tabs(["📊 Chart", "✅ Checks", "📝 How to read", "💰 Position"])
        with k1:
            if px is not None:
                f1 = draw(px, sym)
                f1.update_layout(height=560)
                st.plotly_chart(f1, use_container_width=True, config={"displayModeBar": False})
                st.plotly_chart(compare(px, nifty, sym), use_container_width=True, config={"displayModeBar": False})
        with k2:
            chk = list(features(px, nifty)[1].columns) if px is not None else []
            chips = lambda ks, c: "".join(f'<span class="chip" style="background:{c}26;color:{c};border:1px solid {c}66">{escape(k)}</span>' for k in ks)
            st.markdown("**Passed**")
            st.markdown(chips([k for k in chk if row[k] == "✔"], COL["🟢"]) or "none", unsafe_allow_html=True)
            st.markdown("**Not passed**")
            st.markdown(chips([k for k in chk if row[k] == ""], COL["🔴"]) or "none", unsafe_allow_html=True)
        with k3:
            st.markdown(guide(row))
        with k4:
            per = row["Close"] - row["Stop"]
            if per > 0:
                risk_rs = capital * risk_pct / 100
                qty = int(risk_rs / per)
                val = qty * row["Close"]
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Shares", f"{qty:,}")
                p2.metric("Capital used", f"₹{val:,.0f}", f"{val / capital * 100:.0f}% of capital", delta_color="off")
                p3.metric("Money at risk", f"₹{risk_rs:,.0f}")
                p4.metric("1:2 target", f"₹{row['Close'] + 2 * per:,.2f}")
                if val > 0.20 * capital:
                    st.warning("This position is over 20% of your capital. Consider fewer shares or skip it.")
                st.caption(f"Fifteen losses in a row at this risk would cost about {(1 - (1 - risk_pct / 100) ** 15) * 100:.0f}% of capital, so keep risk small. "
                           "Check the company's results date and recent news before entering; scanners cannot see event risk.")
            else:
                st.info("The stop is not below the price, so no position size can be calculated.")
        st.link_button("Open on TradingView (optional)", "https://in.tradingview.com/chart/?symbol=NSE%3A" + quote(sym, safe=""), use_container_width=True)

with t_acc:
    if sm is None:
        st.info("Not enough past 🟢 signals to test.")
    else:
        a = sm.iloc[0]
        ar = a["Avg R (fixed 1:2 exit)"]
        st.markdown('<div class="kpis">' + kpi("Past signals tested", int(a["Signals"]), "#4f46e5") + kpi("Hit 1:2 target first", f"{a['Hit 1:2 target %']}%", COL["🟢"])
                    + kpi("Stopped out first", f"{a['Stopped out %']}%", COL["🔴"]) + kpi("Avg per trade (after costs)", f"{ar} R", COL["🟢"] if ar > 0 else COL["🔴"])
                    + "</div>", unsafe_allow_html=True)
        (st.success if ar > 0 else st.error)(
            "Positive average result on past data, but it is an estimate, not a promise. Paper trade first." if ar > 0
            else "After costs the fixed 1:2 exit did not make money on past data. Use this scanner as a study tool and paper trade.")
        st.markdown("**By setup**")
        st.dataframe(sm, use_container_width=True, hide_index=True)
        st.markdown("**By year** (big swings between years mean the edge is unreliable)")
        st.dataframe(by_year(bt, cost_pct), use_container_width=True, hide_index=True)
        with st.expander("Method and limits"):
            st.caption("Each past day a stock met today's 🟢 rules, buy at the NEXT day's open, stop at the signal day's 10-day low, gaps below the stop fill at the open, "
                       "costs deducted. Two exits are compared: a fixed 1:2 target (20 days) and riding the trend until a close below the 20 EMA (60 days). "
                       "R = one unit of risk. A 1:2 target breaks even near a 34% hit rate. Only today's Nifty 100 and Midcap 150 companies are tested "
                       "(survivorship bias flatters results), signals cluster in bull markets, and results/news surprises are not modelled.")
        st.download_button("⬇️ Download backtest trades (CSV)", bt.to_csv(index=False), "backtest.csv", use_container_width=True)

with t_guide:
    guide_view()
