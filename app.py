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
    def grab(names, size):
        for i in range(0, len(names), size):
            tick = [n + ".NS" for n in names[i:i + size]]
            try:
                d = yf.download(tick, period="5y", interval="1d", group_by="ticker",
                                auto_adjust=True, threads=True, progress=False)
            except Exception:
                continue
            for t in tick:
                try:
                    x = d[t].dropna()
                except KeyError:
                    continue
                if len(x) > 260:
                    out[t[:-3]] = naive(x)
    grab(list(symbols), 100)
    grab([n for n in symbols if n not in out], 25)
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
    c, v, h, l = x["Close"], x["Volume"], x["High"], x["Low"]
    n = nifty.reindex(x.index).ffill()
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    macd = ema(c, 12) - ema(c, 26)
    adx14, atr = adx_atr(h, l, c)
    f = pd.DataFrame(index=x.index)
    f["Close"], f["RSI"], f["e20"] = c, rsi(c), e20
    f["Level"] = c.shift(1).rolling(60).max()
    f["gap"] = (c / f["Level"] - 1) * 100
    f["fresh"] = c.shift(1) <= f["Level"].shift(1)
    f["vr"] = v / v.shift(1).rolling(50).mean()
    f["rng"] = (h.shift(1).rolling(15).max() - l.shift(1).rolling(15).min()) / c * 100
    f["stop"] = l.rolling(10).min()
    f["risk"] = (c - f["stop"]) / c * 100
    f["ext"] = (c / e20 - 1) * 100
    f["strong"] = (c >= l + 0.75 * (h - l)) & ((h - l) <= 2.5 * atr)
    f["nr7"] = (h - l) <= (h - l).rolling(7).min()
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
    dry = v.shift(1).rolling(10).mean() / v.shift(1).rolling(50).mean() < 0.85
    both = ok["Beats Nifty (3m)"] & ok["Beats Nifty (6m)"]
    f["Quality"] = (np.clip((f["vr"] - 1) / 1.5, 0, 1) * 25 + f["strong"] * 15
                    + np.select([f["rng"] <= 6, f["rng"] <= 10, f["rng"] <= 14], [20, 12, 5], 0) + dry * 10
                    + np.select([adx14 >= 25, adx14 >= 20], [10, 6], 0) + ok["Within 10% of 52w high"] * 10 + both * 10).round()
    conf = f["up"] & (f["ext"] <= 12) & (f["risk"] <= 12) & (f["gap"] > 0) & f["fresh"]
    f["signal"] = np.select([conf & (f["vr"] >= 1.5) & f["strong"], conf], ["🟢", "🟡"], default="")
    return f, ok

def scan(x, nifty):
    f, ok = features(x, nifty)
    t = f.iloc[-1]
    age = next((a for a in range(3) if f["signal"].iloc[-1 - a]), None)
    s = f.iloc[-1 - age] if age is not None else t
    day = ["today", "yesterday", "2 days ago"][age] if age is not None else ""
    since = (t["Close"] / s["Close"] - 1) * 100 if age is not None else np.nan
    risk = (t["Close"] - s["stop"]) / t["Close"] * 100
    flag = ("price jump over 30% (possible split error)" if x["Close"].pct_change().iloc[-60:].abs().max() > .30 else
            "stale data" if (pd.Timestamp.today().normalize() - x.index[-1]).days > 6 else
            "zero volume" if (x["Volume"].iloc[-5:] == 0).any() else "")
    if flag:
        st_ = "🔴 Check data first: " + flag
    elif age is not None:
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
            "Score": int(t["Score"]), "Quality": int(s["Quality"]) if pd.notna(s["Quality"]) else 0, "Data flag": flag, "Close": round(t["Close"], 2), "Breakout level": round(s["Level"], 2),
            "% vs level": round((t["Close"] / s["Level"] - 1) * 100, 1), "Since breakout %": round(since, 1),
            "Volume x": round(s["vr"], 1), "Range %": round(s["rng"], 1), "RSI": round(t["RSI"], 1), "6m return %": round((t["Close"] / x["Close"].iloc[-127] - 1) * 100, 1),
            "3m return %": round((t["Close"] / x["Close"].iloc[-64] - 1) * 100, 1), "NR7": "✔" if t["nr7"] else "",
            "Stop": round(s["stop"], 2), "Risk %": round(risk, 1), "Above 20 EMA %": round(t["ext"], 1),
            "Liquidity (Rs cr)": round((x["Close"] * x["Volume"]).iloc[-20:].mean() / 1e7, 1),
            **{k: ("✔" if b else "") for k, b in ok.iloc[-1].items()}}

def backtest(prices, nifty, horizon=20, trail_days=60):
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
                         "Tight": bool(f["rng"].iloc[i] <= 10), "Quality": f["Quality"].iloc[i]})
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
                    ("Only with a tight base (under 10%)", b["Tight"]), ("Both conditions", b["Green"] & b["Tight"]),
                    ("Quality 70 or more", b["Quality"] >= 70), ("Quality under 70", b["Quality"] < 70)):
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
2. **Volume (middle panel):** {'on the breakout day' if r['Breakout'] else 'today'} was {r['Volume x']}x its 50-day average. A real breakout needs about 1.5x or more.
3. **Base (last 3 weeks):** price moved inside a {r['Range %']}% range. Under 10% is tight and healthy.
4. **Trend (EMA lines):** price should sit above the 50 and 200 EMA. It is {abs(r['Above 20 EMA %'])}% {'above' if r['Above 20 EMA %'] >= 0 else 'below'} the 20 EMA.
5. **RSI (bottom panel): {r['RSI']}.** Healthy uptrends hold between 40 and 80.
6. **Risk:** stop near Rs {r['Stop']} ({r['Risk %']}% below the price).
7. **Valuation:** {pe_t}

**Verdict: {r['Status'][2:]}.** {verdict}"""

# --- NEW: PLOTLY CHARTS THEMED FOR NEON DARK MODE ---
def draw(x, sym):
    c = x["Close"]
    d = pd.DataFrame({"e20": ema(c, 20), "e50": ema(c, 50), "e200": ema(c, 200), "rsi": rsi(c),
                      "vavg": x["Volume"].rolling(50).mean()}).join(x).tail(180)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[.6, .2, .2], vertical_spacing=.03)
    
    # Neon Candlesticks
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                                 name="Price", increasing_line_color="#00E1FF", decreasing_line_color="#FF3366",
                                 increasing_fillcolor="#00E1FF", decreasing_fillcolor="#FF3366"), 1, 1)
    
    # Glowing EMAs
    for k, name, col in (("e20", "EMA 20", "#3b82f6"), ("e50", "EMA 50", "#a855f7"), ("e200", "EMA 200", "#475569")):
        fig.add_trace(go.Scatter(x=d.index, y=d[k], name=name, line=dict(color=col, width=2)), 1, 1)
        
    fig.add_hline(y=c.iloc[-61:-1].max(), line_dash="dash", line_color="#00E1FF",
                  annotation_text="Breakout level", row=1, col=1)
    fig.add_hline(y=x["Low"].iloc[-10:].min(), line_dash="dash", line_color="#FF3366",
                  annotation_text="Stop", row=1, col=1)
                  
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="Volume", marker_color="#1E293B"), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["vavg"], name="50-day avg volume", line=dict(color="#00E1FF", width=1.5)), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["rsi"], name="RSI 14", line=dict(color="#a855f7", width=2)), 3, 1)
    
    for lvl in (40, 80):
        fig.add_hline(y=lvl, line_dash="dot", line_color="#334155", row=3, col=1)
        
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(height=680, title=sym, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    return fig

def compare(x, nifty, sym):
    d = pd.concat([x["Close"], nifty], axis=1, keys=[sym, "Nifty 50"]).dropna().tail(130)
    d = d / d.iloc[0] * 100
    
    # Smooth glowing curves matching the image UI
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d[sym], name=sym, mode="lines", line=dict(color="#00D2FF", width=3, shape="spline")))
    fig.add_trace(go.Scatter(x=d.index, y=d["Nifty 50"], name="Nifty 50", mode="lines", line=dict(color="#334155", width=2, shape="spline")))
    
    fig.update_layout(height=300, title="Relative Strength (Last 6 Months)",
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    return fig

COLS = ["Company Name", "Symbol", "Cap", "Status", "Breakout", "Quality", "Score", "Close", "Breakout level", "% vs level",
        "Since breakout %", "Volume x", "RS rank %", "Range %", "RSI", "Stop", "Risk %", "P/E", "Sector rank", "Data flag", "Industry"]

# --- NEW: COMPLETELY REWRITTEN CSS TO MATCH THE IMAGE ---
CSS = """<style>
/* Base Dark Theme & Abstract Glowing Background */
.stApp {
    background-color: #070A11;
    background-image: 
        radial-gradient(circle at 10% 10%, rgba(0, 210, 255, 0.15) 0%, transparent 45%),
        radial-gradient(circle at 90% 80%, rgba(59, 130, 246, 0.12) 0%, transparent 45%);
    background-attachment: fixed;
    color: #FFFFFF;
    font-family: 'Inter', -apple-system, sans-serif;
}

.stMarkdown, .stCaption, label, p { color: #94A3B8; }
.block-container { padding: 2rem 1rem 6.5rem; max-width: 900px; }

/* Dashboard Header (replaces hero2) */
.hero2 {
    padding: 24px;
    border-radius: 24px;
    background: #111724;
    border: 1px solid #1E2738;
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    margin-bottom: 1.5rem;
}
.hero2 .hi { display: flex; justify-content: space-between; align-items: center; font-weight: 600; color: #94A3B8; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 1px;}
.hero2 .big { font-size: 3.2rem; font-weight: 800; color: #FFFFFF; line-height: 1.1; margin-top: 10px; }
.hero2 .sub { color: #64748B; font-size: 0.85rem; margin-bottom: 12px; }
.mkp { padding: 4px 14px; border-radius: 999px; font-size: 0.75rem; font-weight: 800; color: #FFFFFF; }

/* KPI Grid */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 12px 0 20px; }
.kpi {
    background: #111724;
    border: 1px solid #1E2738;
    border-radius: 20px;
    padding: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
}
.kv { font-size: 1.8rem; font-weight: 800; line-height: 1.2; }
.kl { font-weight: 600; font-size: 0.85rem; color: #E2E8F0; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px;}
.ks { font-size: 0.75rem; color: #64748B; margin-top: 2px;}

/* Status Pills */
.pill { display: inline-block; padding: 4px 12px; border-radius: 8px; font-size: 0.75rem; font-weight: 700; white-space: normal; text-align: center; }

/* Stock Cards */
.scard {
    background: #111724;
    border: 1px solid #1E2738;
    border-radius: 20px;
    padding: 20px;
    margin: 12px 0;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    transition: transform 0.2s ease;
}
.scard:hover { transform: translateY(-2px); border-color: #3b82f6; }
.scard .top { display: flex; justify-content: space-between; gap: 8px; align-items: flex-start; }
.scard b { color: #FFFFFF; font-size: 1.1rem;}
.scard small { display: block; color: #64748B; font-size: 0.8rem; margin-top: 2px;}
.scard .px { margin-top: 10px; font-size: 1.4rem; color: #FFFFFF; font-weight: 700;}
.mini { margin-top: 6px; }

/* Stock Card Internal Grid */
.scard .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 16px; padding-top: 16px; border-top: 1px solid #1E2738;}
.scard .grid div { display: flex; flex-direction: column; font-size: 1rem; color:#fff; font-weight:600;}
.scard .grid span { font-size: 0.7rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px;}

/* Buttons matching the Get Started button in image */
button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg, #0072FF 0%, #00C6FF 100%);
    border: 0;
    border-radius: 12px;
    font-weight: 700;
    color: white;
    padding: 0.5rem 1rem;
    box-shadow: 0 4px 15px rgba(0, 198, 255, 0.4);
}

/* Tabs */
div[data-baseweb="tab-list"] {
    background: #111724;
    border: 1px solid #1E2738;
    border-radius: 16px;
    padding: 8px;
    gap: 4px;
}
div[data-baseweb="tab-list"] button[data-baseweb="tab"] { color: #64748B; border-radius: 12px; }
div[data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #FFFFFF;
    background: #1E293B;
    border: 1px solid #334155;
}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none; }

/* Ring Circular Progress */
.rings { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; margin: 16px 0; }
.ringbox {
    display: flex; align-items: center; gap: 16px; padding: 20px;
    border-radius: 24px; background: #111724; border: 1px solid #1E2738;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
}
.ringbox small { display: block; color: #64748B; margin-top: 4px; }
</style>"""

COL = {"🟢": "#00E1FF", "🟡": "#FBBF24", "🔵": "#3B82F6", "⚪": "#94A3B8", "🟠": "#F97316", "🔴": "#EF4444"}

PX = {}

def pill(text, color):
    return f'<span class="pill" style="background:{color}1A;color:{color};border:1px solid {color}40">{escape(str(text))}</span>'

def kpi(label, value, color, sub=""):
    return (f'<div class="kpi" style="border-top:4px solid {color}"><div class="kv" style="color:{color}">{value}</div>'
            f'<div class="kl">{label}</div><div class="ks">{sub}</div></div>')

def fmt(v, suf="", nd=1):
    return "–" if v is None or pd.isna(v) else f"{v:,.{nd}f}{suf}"

def spark(vals, color="#00D2FF", w=320, h=90):
    v = np.asarray(vals, float)
    ys = h - 8 - (v - v.min()) / (v.max() - v.min() + 1e-9) * (h - 24)
    xs = np.linspace(4, w - 8, len(v))
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))
    gid = "g" + color[1:]
    return (f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:auto;display:block"><defs><linearGradient id="{gid}" x1="0" x2="0" y1="0" y2="1">'
            f'<stop offset="0" stop-color="{color}" stop-opacity=".4"/><stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
            f'<polygon points="4,{h} {pts} {xs[-1]:.1f},{h}" fill="url(#{gid})"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round" style="filter:drop-shadow(0 0 8px {color})"/>'
            f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="6" fill="#fff" style="filter:drop-shadow(0 0 6px #fff)"/></svg>')

def ring(pct, title, sub, color="#00D2FF"):
    c = 2 * 3.14159 * 52
    return (f'<div class="ringbox"><svg viewBox="0 0 130 130" width="120" height="120" style="flex:none"><circle cx="65" cy="65" r="52" fill="none" stroke="#1E293B" stroke-width="12"/>'
            f'<circle cx="65" cy="65" r="52" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round" stroke-dasharray="{c * pct / 100:.1f} {c:.1f}" '
            f'transform="rotate(-90 65 65)" style="filter:drop-shadow(0 0 8px {color})"/>'
            f'<circle cx="{65 + 52 * np.cos(np.radians(pct * 3.6 - 90)):.1f}" cy="{65 + 52 * np.sin(np.radians(pct * 3.6 - 90)):.1f}" r="8" fill="#fff" style="filter:drop-shadow(0 0 6px #fff)"/>'
            f'<text x="65" y="73" text-anchor="middle" fill="#fff" font-size="28" font-weight="800">{pct:.0f}%</text></svg>'
            f'<div><b style="color:#fff">{title}</b><small>{sub}</small></div></div>')

def dark(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    # Style gridlines to match deep dark UI
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#1E293B')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#1E293B')
    return fig

def card(r):
    c = COL[r["Status"][0]]
    x = PX.get(r["Symbol"])
    chg = sp = ""
    if x is not None and len(x) > 41:
        cl = x["Close"]
        d = (cl.iloc[-1] / cl.iloc[-2] - 1) * 100
        chg = f'<span style="color:{"#00E1FF" if d >= 0 else "#FF3366"};font-weight:700;font-size:.9rem">{"▲" if d >= 0 else "▼"} {abs(d):.1f}%</span>'
        sp = f'<div class="mini">{spark(cl.tail(40).values, c, 320, 56)}</div>'
    m = [("Level", f"₹{r['Breakout level']:,.2f}"), ("vs level", fmt(r["% vs level"], "%")), ("Volume", fmt(r["Volume x"], "x")),
         ("Quality", f"{int(r['Quality'])}/100"), ("RS rank", fmt(r["RS rank %"], nd=0)), ("Risk", fmt(r["Risk %"], "%")),
         ("P/E", fmt(r.get("P/E"))), ("Score", f"{int(r['Score'])}/9")]
    cells = "".join(f"<div><b>{v}</b><span>{k}</span></div>" for k, v in m)
    return (f'<div class="scard" style="border-left:4px solid {c}"><div class="top"><div><b>{escape(str(r["Company Name"]))}</b>'
            f'<small>{escape(str(r["Symbol"]))} · {r["Cap"]} · {escape(str(r["Industry"]))}</small></div>{pill(r["Status"][2:], c)}</div>'
            f'<div class="px">₹{r["Close"]:,.2f} {chg}</div>{sp}<div class="grid">{cells}</div></div>')

def tint(v):
    c = COL.get(str(v)[:1])
    return f"background-color:{c}1A;font-weight:600;color:{c}" if c else ""

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
    st.markdown("".join(f'<div class="scard" style="border-left:4px solid {COL[i]}"><b>{i} {t}</b><small>{d}</small></div>' for i, t, d in items),
                unsafe_allow_html=True)
    with st.expander("Columns and terms"):
        st.markdown("**Quality (0-100)** ranks a signal on volume surge, strong candle, tight base... (rest of guide omitted for brevity)")

st.markdown(CSS, unsafe_allow_html=True)
try:
    uni = load_universe()
except Exception:
    up = st.file_uploader("Could not fetch the NSE lists. Upload any NSE index CSV", type="csv")
    if up is None:
        st.stop()
    uni = pd.read_csv(up)[["Company Name", "Industry", "Symbol"]].assign(Cap="Large/Mid")

nifty = load_nifty()
n_last, d50, d200 = nifty.iloc[-1], nifty.rolling(50).mean().iloc[-1], nifty.rolling(200).mean().iloc[-1]
good = n_last > d50 and n_last > d200
mk = ("#00E1FF", "MARKET GREEN") if good else ("#F97316", "MARKET CAUTION")
d1 = (n_last / nifty.iloc[-2] - 1) * 100
st.markdown(f'<div class="hero2"><div class="hi"><span>📈 Market pulse</span><span class="mkp" style="background:{mk[0]}">{mk[1]}</span></div>'
            f'<div class="big">{n_last:,.0f} <span style="font-size:1.2rem;color:{"#00E1FF" if d1 >= 0 else "#FF3366"}">{"▲" if d1 >= 0 else "▼"} {abs(d1):.2f}%</span></div>'
            f'<div class="sub">Nifty 50 · {(n_last / d50 - 1) * 100:+.1f}% vs 50 DMA · {(n_last / d200 - 1) * 100:+.1f}% vs 200 DMA</div>'
            f'{spark(nifty.tail(90).values, "#00D2FF", 320, 90)}</div>', unsafe_allow_html=True)

b1, b2 = st.columns([1, 2])
run = b1.button("🔄 Run scan", type="primary", use_container_width=True)
b2.caption(f"Last scan: {st.session_state['ts']}. Yahoo data, about 15 min delayed." if "ts" in st.session_state else "Best after 4 PM IST.")

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

@st.cache_data(ttl=1800, show_spinner="Scanning and backtesting 5 years of history...")
def run_all(universe: pd.DataFrame):
    prices = load_prices(tuple(universe["Symbol"]))
    nf = load_nifty()
    if not prices:
        raise RuntimeError("no price data came back from Yahoo Finance")
    r0 = pd.DataFrame([{"Symbol": k, **scan(x, nf)} for k, x in prices.items()]).merge(universe, on="Symbol")
    r0["RS rank %"] = (r0["6m return %"].rank(pct=True) * 100).round()
    sec = r0.groupby("Industry")["3m return %"].median()
    r0["Sector 3m %"] = r0["Industry"].map(sec).round(1)
    r0["Sector rank"] = r0["Industry"].map(sec.rank(ascending=False)).round().astype("Int64")
    return r0, backtest(prices, nf)

if run:
    try:
        st.session_state["res"], st.session_state["bt"] = run_all(uni)
        st.session_state["ts"] = datetime.now().strftime("%d %b, %H:%M")
    except Exception as e:
        st.error(f"Scan failed ({e}). Wait a few minutes and try again.")
        st.stop()
    st.rerun()

res = st.session_state.get("res")
if res is None:
    st.markdown('<div class="scard" style="border-left:4px solid #00D2FF"><b>👋 Start here</b><small>1) Tap <b>Run scan</b> above.</small></div>', unsafe_allow_html=True)
    st.stop()

df = res[(res["Score"] >= min_score) & (res["Liquidity (Rs cr)"] >= min_liq)]
df = df[df["Cap"].isin(caps + ["Large/Mid"])]
if sectors:
    df = df[df["Industry"].isin(sectors)]
if top_sec:
    df = df[(df["Sector rank"] <= top_sec).fillna(False)]
df = df.sort_values(["Days ago", "Quality", "Score", "Volume x"], ascending=[True, False, False, False])
focus = df[df["Status"].str[0].isin(["🟢", "🟡", "🔵", "⚪"]) | (df["Breakout"] != "")].head(40)
with st.spinner("Fetching P/E for the top candidates..."):
    fund = {s: fundamentals(s) for s in focus["Symbol"]}
df = df.assign(**{k: df["Symbol"].map(lambda s, k=k: fund.get(s, {}).get(k)) for k in ("P/E", "Debt/Equity", "Profit growth %")})
S0 = df["Status"].str[0]
buckets = {"🟢 Breakouts": df["Breakout"] != "", "🔵 Watchlist": S0.isin(["🔵", "⚪"]) & (df["Breakout"] == ""),
           "🟠 Don't chase": (S0 == "🟠") & (df["Breakout"] == ""), "🔴 Avoid": S0 == "🔴", "All": S0.notna()}
bt = st.session_state.get("bt")
sm = summarise(bt, cost_pct) if bt is not None and not bt.empty else None

PX = load_prices(tuple(uni["Symbol"]))
t_home, t_scan, t_study, t_acc, t_guide = st.tabs(["🏠 Home", "🔍 Scan", "📈 Study", "🎯 Accuracy", "📘 Guide"])

with t_home:
    upc = "Uptrend (price>50>200 EMA, 200 rising)"
    nup = int((res[upc] == "✔").sum())
    rings = ring(nup / len(res) * 100, "Market breadth", f"{nup} of {len(res)} stocks in an uptrend")
    if sm is not None:
        a = sm.iloc[0]
        rings += ring(a["Hit 1:2 target %"], "Past hit rate", f"{int(a['Signals'])} past breakouts", "#a855f7")
    st.markdown(f'<div class="rings">{rings}</div>', unsafe_allow_html=True)
    cnt = {k: int(m.sum()) for k, m in buckets.items()}
    cards = [kpi("Breakouts (3 days)", cnt["🟢 Breakouts"], COL["🟢"]), kpi("Watchlist", cnt["🔵 Watchlist"], COL["🔵"]),
             kpi("Don't chase", cnt["🟠 Don't chase"], COL["🟠"]), kpi("Avoid", cnt["🔴 Avoid"], COL["🔴"])]
    if sm is not None:
        cards.append(kpi("Avg per trade", f"{a['Avg R (fixed 1:2 exit)']} R", COL["🟢"] if a["Avg R (fixed 1:2 exit)"] > 0 else COL["🔴"], "after costs"))
    st.markdown('<div class="kpis">' + "".join(cards) + "</div>", unsafe_allow_html=True)
    for title, key in (("🟢 Latest breakouts", "🟢 Breakouts"), ("🔵 Top watchlist", "🔵 Watchlist")):
        d = df[buckets[key]].head(5)
        st.markdown(f"#### {title}")
        if d.empty:
            st.info("None right now.")
        else:
            st.markdown("".join(card(r) for _, r in d.iterrows()), unsafe_allow_html=True)
    secs = res.groupby("Industry")["3m return %"].median().sort_values(ascending=False)
    top = pd.concat([secs.head(6), secs.tail(3)]).drop_duplicates()
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=[COL["🟢"] if v >= 0 else COL["🔴"] for v in top.values]))
    fig.update_layout(height=340, title="Sector strength: median 3-month return %", yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(dark(fig), use_container_width=True, config={"displayModeBar": False})

with t_scan:
    pick = st.radio("List", [f"{k} ({int(m.sum())})" for k, m in buckets.items()], horizontal=True, label_visibility="collapsed")
    name = list(buckets)[[f"{k} ({int(m.sum())})" for k, m in buckets.items()].index(pick)]
    d = df[buckets[name]]
    if d.empty:
        st.info("No stocks in this list with your filters.")
    elif st.toggle("Table view (all rows)", False):
        more = st.toggle("Show all columns", False)
        cols = COLS if more else ["Company Name", "Status", "Breakout", "Quality", "Score", "Close", "% vs level", "Volume x", "RS rank %", "Risk %", "P/E"]
        st.dataframe(styled(d[cols]), use_container_width=True, hide_index=True, height=520)
    else:
        st.markdown("".join(card(r) for _, r in d.head(30).iterrows()), unsafe_allow_html=True)

with t_study:
    if df.empty:
        st.info("No stocks match your filters.")
    else:
        order = list(focus["Symbol"]) + [s for s in df["Symbol"] if s not in set(focus["Symbol"])]
        names = dict(zip(df["Symbol"], df["Company Name"]))
        sym = st.selectbox("Choose a stock", order, format_func=lambda s: f"{names[s]} ({s})")
        row = df[df["Symbol"] == sym].iloc[0]
        px = load_prices(tuple(uni["Symbol"])).get(sym)
        st.markdown(card(row), unsafe_allow_html=True)
        k1, k2, k3, k4 = st.tabs(["📊 Chart", "✅ Checks", "📝 How to read", "💰 Position"])
        with k1:
            if px is not None:
                f1 = draw(px, sym)
                dark(f1)
                st.plotly_chart(f1, use_container_width=True, config={"displayModeBar": False})
                st.plotly_chart(dark(compare(px, nifty, sym)), use_container_width=True, config={"displayModeBar": False})
        with k2:
            chk = list(features(px, nifty)[1].columns) if px is not None else []
            chips = lambda ks, c: "".join(f'<span class="chip" style="background:{c}1A;color:{c};border:1px solid {c}40">{escape(k)}</span>' for k in ks)
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
                p2.metric("Capital used", f"₹{val:,.0f}")
                p3.metric("Money at risk", f"₹{risk_rs:,.0f}")
                p4.metric("1:2 target", f"₹{row['Close'] + 2 * per:,.2f}")

with t_acc:
    if sm is None:
        st.info("Not enough past 🟢 signals to test.")
    else:
        st.dataframe(sm, use_container_width=True, hide_index=True)

with t_guide:
    guide_view()
