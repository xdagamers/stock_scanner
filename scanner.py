import io
import math
import requests
import numpy as np
import pandas as pd
import yfinance as yf

NSE_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv"
FALLBACK_URL = (
    "https://huggingface.co/spaces/Invicto69/Algo_Trading_Dashboard_streamlit/"
    "raw/main/data/ind_nifty200list.csv"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept": "text/csv,text/plain,*/*",
}

def load_universe():
    """Load NIFTY 200 constituents. Official NSE first, fallback mirror second."""
    for url in (NSE_URL, FALLBACK_URL):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            symbol_col = next((c for c in df.columns if c.strip().lower() == "symbol"), None)
            if symbol_col is None:
                continue
            name_col = next(
                (c for c in df.columns if c.strip().lower() in {"company name", "company_name"}),
                None,
            )
            out = pd.DataFrame({
                "symbol": df[symbol_col].astype(str).str.strip(),
                "company_name": (
                    df[name_col].astype(str).str.strip()
                    if name_col else df[symbol_col].astype(str).str.strip()
                ),
            })
            out = out[out["symbol"].ne("") & out["symbol"].ne("nan")]
            out = out[~out["symbol"].str.upper().isin(["DUMMYITC"])]
            return out.drop_duplicates("symbol").reset_index(drop=True)
        except Exception:
            pass
    raise RuntimeError("Could not load NIFTY 200 constituents.")

def yf_symbol(symbol):
    return symbol.replace("&", "%26") + ".NS"

def fetch_history(symbol, period="2y"):
    """Daily EOD OHLCV from Yahoo Finance."""
    ticker = yf_symbol(symbol)
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        needed = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in needed):
            return None
        df = df[needed].dropna()
        if len(df) < 220:
            return None
        return df
    except Exception:
        return None

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close):
    fast = ema(close, 12)
    slow = ema(close, 26)
    line = fast - slow
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return line, signal, hist

def atr(df, n=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def adx(df, n=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_v = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_v.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_v.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_v = dx.ewm(alpha=1/n, adjust=False).mean()
    return adx_v, plus_di, minus_di

def candle_pattern(df):
    if len(df) < 3:
        return "None"
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    def body(x): return abs(x["Close"] - x["Open"])
    def rng(x): return max(x["High"] - x["Low"], 1e-9)

    # Current bullish engulfing of previous body
    if b["Close"] < b["Open"] and c["Close"] > c["Open"]:
        if c["Open"] <= b["Close"] and c["Close"] >= b["Open"]:
            return "Bullish Engulfing"

    # Hammer
    lower = min(c["Open"], c["Close"]) - c["Low"]
    upper = c["High"] - max(c["Open"], c["Close"])
    if lower >= 2 * body(c) and upper <= body(c) * 1.2:
        return "Hammer"

    # Bullish Marubozu
    if c["Close"] > c["Open"] and body(c) / rng(c) > 0.75:
        return "Bullish Marubozu"

    # Morning star approximation
    if a["Close"] < a["Open"] and body(b) < body(a) * 0.5 and c["Close"] > c["Open"]:
        if c["Close"] > (a["Open"] + a["Close"]) / 2:
            return "Morning Star"

    # Piercing line
    if b["Close"] < b["Open"] and c["Close"] > c["Open"]:
        if c["Open"] < b["Low"] and c["Close"] > (b["Open"] + b["Close"]) / 2:
            return "Piercing Line"

    # Three white soldiers
    if all(x["Close"] > x["Open"] for x in [a, b, c]) and a["Close"] < b["Close"] < c["Close"]:
        return "Three White Soldiers"

    # Inside bar
    if c["High"] < b["High"] and c["Low"] > b["Low"]:
        return "Inside Bar"

    return "None"

def resistance_level(df, lookback=150):
    """
    Zone-based resistance:
    - uses only bars before the latest bar
    - identifies local swing highs
    - clusters highs within ~1.25%
    - scores clusters by touches, recency and distance
    Returns zone_top, touches, recency_days.
    """
    hist = df.iloc[:-1].tail(lookback).copy()
    current = float(df["Close"].iloc[-1])
    if len(hist) < 30:
        return np.nan, 0, 999

    h = hist["High"].astype(float)
    swing = (h > h.shift(1)) & (h >= h.shift(-1))
    highs = h[swing].dropna()

    if len(highs) == 0:
        return float(h.max()), 1, lookback

    levels = sorted(highs.tolist())
    clusters = []
    tolerance = 0.0125

    for level in levels:
        placed = False
        for cluster in clusters:
            center = np.mean(cluster)
            if abs(level - center) / center <= tolerance:
                cluster.append(level)
                placed = True
                break
        if not placed:
            clusters.append([level])

    candidates = []
    for cluster in clusters:
        center = float(np.mean(cluster))
        top = float(max(cluster))
        touch_count = len(cluster)

        # recency = bars since most recent member of the zone
        recencies = []
        for level in cluster:
            positions = np.where(np.isclose(h.values, level, rtol=1e-8, atol=1e-8))[0]
            if len(positions):
                recencies.append(len(hist) - int(positions[-1]))
        recency = min(recencies) if recencies else lookback

        distance = (top - current) / current * 100
        # Allow slightly pierced resistance; strong zones can be just below price.
        if distance >= -2.0:
            distance_penalty = abs(distance)
            strength = touch_count * 4 + max(0, 4 - recency / 30)
            candidates.append((distance_penalty, -strength, top, touch_count, recency))

    if not candidates:
        top = float(h.max())
        return top, 1, lookback

    # First prioritize a nearby zone; within similar distance prefer stronger zone.
    candidates.sort(key=lambda x: (x[0], x[1]))
    _, _, top, touches, recency = candidates[0]
    return top, touches, recency

def false_breakout_info(df, lookback=20, tolerance=0.0025):
    """
    Detects recent false breakouts:
    price moved above a prior resistance by at least tolerance,
    then closed back below that resistance within 1-3 bars.
    """
    if len(df) < lookback + 5:
        return False, 0

    recent = df.tail(lookback + 4).copy()
    closes = recent["Close"].astype(float)
    highs = recent["High"].astype(float)

    for i in range(3, len(recent)):
        prior = recent.iloc[:i]
        resistance = float(prior["High"].max())
        bar = recent.iloc[i]
        if float(bar["High"]) > resistance * (1 + tolerance):
            for j in range(i + 1, min(i + 4, len(recent))):
                if float(recent.iloc[j]["Close"]) < resistance:
                    return True, (len(recent) - 1) - i
    return False, 0

def market_strength(df):
    close = df["Close"]
    if len(close) < 65:
        return 0.0, "Unknown"
    ret20 = float(close.iloc[-1] / close.iloc[-21] - 1) * 100
    ret60 = float(close.iloc[-1] / close.iloc[-61] - 1) * 100
    score = 0
    if ret20 > 0: score += 1
    if ret60 > 0: score += 1
    label = "Strong" if score == 2 else "Neutral" if score == 1 else "Weak"
    return (ret20 + ret60) / 2, label


def fetch_index_history(symbol, period="2y"):
    try:
        df = yf.download(
            symbol, period=period, interval="1d",
            auto_adjust=True, progress=False, threads=False
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception:
        return None

def relative_strength_vs_index(stock_df, index_df, window=60):
    if index_df is None or len(stock_df) <= window or len(index_df) <= window:
        return 0.0
    n = min(len(stock_df), len(index_df))
    a = float(stock_df["Close"].iloc[-1] / stock_df["Close"].iloc[-window-1] - 1)
    b = float(index_df["Close"].iloc[-1] / index_df["Close"].iloc[-window-1] - 1)
    return (a - b) * 100


DEFAULT_WEIGHTS = {
    "proximity": 20,
    "resistance": 15,
    "volume": 15,
    "rsi": 10,
    "macd": 10,
    "adx": 10,
    "ma": 10,
    "candle": 5,
    "relative_strength": 5,
}

def component_scores(df, nifty_df=None, sector_df=None):
    close = df["Close"]
    volume = df["Volume"].replace(0, np.nan)
    price = float(close.iloc[-1])

    breakout, touches, recency = resistance_level(df)
    if not np.isfinite(breakout) or breakout <= 0:
        return None

    distance = (breakout - price) / breakout * 100
    rsi_s = rsi(close)
    rsi_v = float(rsi_s.iloc[-1])

    macd_line, signal, hist = macd(close)
    macd_v = float(macd_line.iloc[-1])
    signal_v = float(signal.iloc[-1])
    macd_hist = float(hist.iloc[-1])
    macd_hist_prev = float(hist.iloc[-2])

    adx_s, plus_di, minus_di = adx(df)
    adx_v = float(adx_s.iloc[-1])
    plus_v = float(plus_di.iloc[-1])
    minus_v = float(minus_di.iloc[-1])

    e20 = float(ema(close, 20).iloc[-1])
    e50 = float(ema(close, 50).iloc[-1])
    e200 = float(close.rolling(200).mean().iloc[-1])

    rvol = float(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1])
    candle = candle_pattern(df)
    ret60 = float(close.iloc[-1] / close.iloc[-61] - 1) * 100 if len(close) > 61 else 0.0

    nifty_rs = relative_strength_vs_index(df, nifty_df, 60) if nifty_df is not None else 0.0
    sector_rs = relative_strength_vs_index(df, sector_df, 60) if sector_df is not None else 0.0
    false_breakout, false_breakout_age = false_breakout_info(df)

    # Normalized component scores are percentages of their maximum weight.
    if distance <= 1: proximity = 1.00
    elif distance <= 2: proximity = .90
    elif distance <= 3: proximity = .80
    elif distance <= 5: proximity = .65
    elif distance <= 7: proximity = .45
    elif distance <= 10: proximity = .25
    else: proximity = 0.0

    resistance = min(1.0, touches / 4)
    if recency <= 35:
        resistance = min(1.0, resistance + .10)

    if rvol >= 3: volume_s = 1.00
    elif rvol >= 2: volume_s = .87
    elif rvol >= 1.5: volume_s = .67
    elif rvol >= 1.2: volume_s = .47
    elif rvol >= 1.0: volume_s = .33
    elif rvol >= .8: volume_s = .20
    else: volume_s = 0.0

    if 55 <= rsi_v <= 68: rsi_scor = .80
    elif 50 <= rsi_v < 55 or 68 < rsi_v <= 72: rsi_scor = .60
    elif 45 <= rsi_v < 50: rsi_scor = .40
    elif rsi_v > 72: rsi_scor = .30
    else: rsi_scor = .20
    if rsi_s.iloc[-1] > rsi_s.iloc[-3]:
        rsi_scor = min(1.0, rsi_scor + .20)

    macd_scor = 0
    if macd_v > signal_v: macd_scor += .50
    if macd_hist > 0: macd_scor += .20
    if macd_hist > macd_hist_prev: macd_scor += .20
    if macd_v > 0: macd_scor += .10

    adx_scor = .10 if adx_v < 15 else .30 if adx_v < 20 else .50 if adx_v < 25 else .70 if adx_v < 35 else .80
    if plus_v > minus_v:
        adx_scor = min(1.0, adx_scor + .20)

    ma_scor = (
        (0.30 if price > e20 else 0) +
        (0.30 if e20 > e50 else 0) +
        (0.20 if e50 > e200 else 0) +
        (0.20 if price > e200 else 0)
    )

    candle_scor = {
        "Bullish Engulfing": 1.00,
        "Bullish Marubozu": 1.00,
        "Morning Star": 1.00,
        "Three White Soldiers": 1.00,
        "Hammer": .80,
        "Piercing Line": .80,
        "Inside Bar": .60,
        "None": 0.0,
    }.get(candle, 0.0)

    # Relative strength rewards outperformance, not absolute return alone.
    rs_base = 0.0
    if nifty_rs >= 10: rs_base = .60
    elif nifty_rs >= 5: rs_base = .40
    elif nifty_rs >= 0: rs_base = .20
    if sector_rs >= 5: rs_base = min(1.0, rs_base + .40)
    elif sector_rs >= 0: rs_base = min(1.0, rs_base + .20)

    return {
        "price": price, "breakout_level": float(breakout), "distance_pct": float(distance),
        "touches": touches, "recency": recency,
        "proximity": proximity, "resistance": resistance, "volume": volume_s,
        "rsi": rsi_scor, "macd": min(1.0, macd_scor), "adx": min(1.0, adx_scor),
        "ma": ma_scor, "candle": candle_scor, "relative_strength": rs_base,
        "rsi_value": rsi_v, "macd_value": macd_v, "adx_value": adx_v, "rvol": rvol,
        "ema_structure": ("Bullish" if price > e20 > e50 > e200 else "Mixed" if price > e200 else "Bearish"),
        "candlestick": candle, "relative_strength_60": ret60,
        "nifty_relative_strength": nifty_rs, "sector_relative_strength": sector_rs,
        "false_breakout": bool(false_breakout), "false_breakout_age": int(false_breakout_age),
    }

def score_from_components(c, weights=None):
    weights = weights or DEFAULT_WEIGHTS
    raw = sum(c[k] * weights[k] for k in weights)
    # Explicit false-breakout penalty. This remains independent of weight tuning.
    if c["false_breakout"]:
        raw = max(0.0, raw - 8.0)
    return min(100.0, max(0.0, raw))

def score_stock(df, nifty_df=None, sector_df=None, weights=None):
    c = component_scores(df, nifty_df=nifty_df, sector_df=sector_df)
    if c is None:
        return None

    score = score_from_components(c, weights)
    rating = max(1, min(10, int(np.floor(score / 10))))

    distance = c["distance_pct"]
    if distance < 0:
        status = "BREAKOUT CONFIRMED"
    elif distance <= 1:
        status = "VERY NEAR"
    elif distance <= 3:
        status = "NEAR BREAKOUT"
    elif distance <= 5:
        status = "DEVELOPING"
    elif distance <= 10:
        status = "WATCH"
    else:
        status = "WEAK"

    reasons = []
    if c["proximity"] >= .8: reasons.append("very close to resistance")
    if c["touches"] >= 3: reasons.append(f"{c['touches']} resistance touches")
    if c["rvol"] >= 1.5: reasons.append("above-average volume")
    if c["rsi_value"] >= 55 and c["rsi_value"] <= 72: reasons.append("supportive RSI")
    if c["ema_structure"] == "Bullish": reasons.append("bullish moving-average structure")
    if c["candlestick"] != "None": reasons.append(c["candlestick"])
    if c["nifty_relative_strength"] > 0: reasons.append("outperforming NIFTY")
    if c["false_breakout"]: reasons.append("recent false-breakout risk")

    explanation = (
        f"Score {score:.0f}/100. " +
        ("; ".join(reasons) if reasons else "limited positive confirmation") +
        ". Score is a technical setup measure, not a probability or price prediction."
    )

    return {
        "price": c["price"], "breakout_level": c["breakout_level"],
        "distance_pct": c["distance_pct"], "score": score, "rating": rating,
        "status": status, "rsi": c["rsi_value"], "macd": c["macd_value"],
        "adx": c["adx_value"], "rvol": c["rvol"],
        "ema_structure": c["ema_structure"], "candlestick": c["candlestick"],
        "relative_strength": c["relative_strength_60"],
        "nifty_relative_strength": c["nifty_relative_strength"],
        "sector_relative_strength": c["sector_relative_strength"],
        "false_breakout": c["false_breakout"],
        "false_breakout_age": c["false_breakout_age"],
        "resistance_touches": c["touches"],
        "explanation": explanation,
    }

def _backtest_with_weights(df, weights, min_rating=7, hold_days=10):
    trades = []
    min_bars = 220
    for i in range(min_bars, len(df) - hold_days - 1):
        hist = df.iloc[:i+1].copy()
        row = score_stock(hist, weights=weights)
        if row is None or row["distance_pct"] < -1 or row["distance_pct"] > 10 or row["rating"] < min_rating:
            continue

        entry = float(hist["Close"].iloc[-1])
        future = df.iloc[i+1:i+1+hold_days]
        level = float(row["breakout_level"])
        future_high = float(future["High"].max())
        final_close = float(future["Close"].iloc[-1])

        trades.append({
            "date": hist.index[-1],
            "breakout_hit": bool(future_high >= level),
            "final_return_pct": (final_close / entry - 1) * 100,
            "max_return_pct": (future_high / entry - 1) * 100,
            "rating": row["rating"],
        })
    return pd.DataFrame(trades)

def optimize_weights(symbols, period="5y", min_rating=7, hold_days=10, max_symbols=12):
    """Tune weights efficiently: each stock is walked once per history; candidates are scored vectorially."""
    symbols = list(symbols)[:max_symbols]
    candidates = [
        DEFAULT_WEIGHTS,
        {"proximity":25,"resistance":15,"volume":15,"rsi":10,"macd":8,"adx":7,"ma":10,"candle":5,"relative_strength":5},
        {"proximity":20,"resistance":20,"volume":15,"rsi":8,"macd":8,"adx":7,"ma":10,"candle":5,"relative_strength":7},
        {"proximity":20,"resistance":15,"volume":20,"rsi":8,"macd":8,"adx":7,"ma":10,"candle":5,"relative_strength":7},
        {"proximity":20,"resistance":15,"volume":15,"rsi":12,"macd":10,"adx":8,"ma":10,"candle":5,"relative_strength":5},
        {"proximity":20,"resistance":15,"volume":15,"rsi":8,"macd":12,"adx":10,"ma":10,"candle":5,"relative_strength":5},
        {"proximity":20,"resistance":15,"volume":15,"rsi":8,"macd":8,"adx":12,"ma":12,"candle":5,"relative_strength":5},
        {"proximity":25,"resistance":20,"volume":15,"rsi":8,"macd":7,"adx":5,"ma":10,"candle":5,"relative_strength":5},
    ]

    histories = {}
    for symbol in symbols:
        df = fetch_history(symbol, period)
        if df is not None:
            histories[symbol] = df

    # Build the historical feature rows only once. Weight candidates then reuse them.
    feature_rows = []
    keys = list(DEFAULT_WEIGHTS.keys())
    for symbol, df in histories.items():
        min_bars = 220
        for i in range(min_bars, len(df) - hold_days - 1):
            hist = df.iloc[:i+1].copy()
            c = component_scores(hist)
            if c is None or c["distance_pct"] < -1 or c["distance_pct"] > 10:
                continue
            future = df.iloc[i+1:i+1+hold_days]
            entry = float(hist["Close"].iloc[-1])
            level = float(c["breakout_level"])
            feature_rows.append({
                **{k: c[k] for k in keys},
                "date": hist.index[-1],
                "breakout_hit": bool(float(future["High"].max()) >= level),
                "final_return_pct": (float(future["Close"].iloc[-1]) / entry - 1) * 100,
                "max_return_pct": (float(future["High"].max()) / entry - 1) * 100,
                "false_breakout": bool(c["false_breakout"]),
            })

    if not feature_rows:
        return pd.DataFrame(), DEFAULT_WEIGHTS

    features = pd.DataFrame(feature_rows)
    rows = []
    for ci, weights in enumerate(candidates, 1):
        score = sum(features[k] * weights[k] for k in keys)
        score = score.where(~features["false_breakout"], score - 8).clip(0, 100)
        rating = np.floor(score / 10).clip(1, 10)
        selected = features.loc[rating >= min_rating].copy()
        if selected.empty:
            continue
        hit = selected["breakout_hit"].mean() * 100
        avg_ret = selected["final_return_pct"].mean()
        median_ret = selected["final_return_pct"].median()
        avg_max = selected["max_return_pct"].mean()
        sample_bonus = min(1.0, len(selected) / 100)
        objective = (0.55 * hit) + (0.35 * max(-10, min(10, avg_ret)) * 5) + (10 * sample_bonus)
        rows.append({
            "candidate": ci, "signals": len(selected),
            "breakout_hit_rate_pct": hit, "avg_final_return_pct": avg_ret,
            "median_final_return_pct": median_ret, "avg_max_return_pct": avg_max,
            "objective": objective, "weights": weights,
        })

    if not rows:
        return pd.DataFrame(), DEFAULT_WEIGHTS
    result = pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)
    return result, result.iloc[0]["weights"]
def scan_single(symbol, max_distance=10.0, period="5y", nifty_df=None):
    """Fetches and scores a single stock."""
    df = fetch_history(symbol, period)
    if df is None or len(df) < 60:
        return None
        
    result = score_stock(df, nifty_df=nifty_df)
    if result is None:
        return None
        
    # Only return stocks within the user's distance threshold
    if result["distance_pct"] <= max_distance:
        result["symbol"] = symbol
        return result
        
    return None

def scan_universe(max_distance=10.0, period="5y"):
    """Scans the entire NIFTY universe and returns a sorted DataFrame."""
    symbols = load_universe()
    
    # Pre-fetch NIFTY 50 index data to calculate relative strength
    nifty_df = None
    try:
        nifty_df = fetch_history("^NSEI", period)
    except Exception:
        pass
        
    results = []
    for sym in symbols:
        row = scan_single(sym, max_distance=max_distance, period=period, nifty_df=nifty_df)
        if row is not None:
            results.append(row)
            
    if not results:
        import pandas as pd
        return pd.DataFrame()
        
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Prioritize the strongest setups (highest rating), then those closest to breaking out
    df = df.sort_values(by=["rating", "distance_pct"], ascending=[False, True]).reset_index(drop=True)
    return df
