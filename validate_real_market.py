"""Real-market validation harness for the NIFTY 200 breakout scanner.

Run locally where internet access is available:
    python validate_real_market.py --years 5 --hold-days 5,10,20

It uses the current NSE NIFTY 200 universe and Yahoo Finance EOD history,
then produces validation CSVs. This is research/validation, not investment advice.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

import scanner


def download_batch(symbols, years=5):
    tickers = [scanner.yf_symbol(s) for s in symbols]
    data = yf.download(
        tickers,
        period=f"{years}y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
        multi_level_index=True,
    )
    histories = {}
    for symbol, ticker in zip(symbols, tickers):
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker not in data.columns.get_level_values(0):
                    continue
                df = data[ticker].copy()
            else:
                df = data.copy()
            needed = ["Open", "High", "Low", "Close", "Volume"]
            if not all(c in df.columns for c in needed):
                continue
            df = df[needed].dropna()
            if len(df) >= 220:
                histories[symbol] = df
        except Exception:
            continue
    return histories


def validate_stock(symbol, df, hold_days):
    rows = []
    min_bars = 220
    for i in range(min_bars, len(df) - max(hold_days) - 1):
        hist = df.iloc[:i + 1]
        c = scanner.component_scores(hist)
        if c is None or c["distance_pct"] < -1 or c["distance_pct"] > 10:
            continue
        score = scanner.score_from_components(c)
        rating = max(1, min(10, int(np.floor(score / 10))))
        entry = float(hist["Close"].iloc[-1])
        row = {
            "symbol": symbol,
            "date": hist.index[-1],
            "score": score,
            "rating": rating,
            "distance_pct": c["distance_pct"],
            "rvol": c["rvol"],
            "adx": c["adx_value"],
            "rsi": c["rsi_value"],
            "false_breakout": c["false_breakout"],
        }
        for h in hold_days:
            future = df.iloc[i + 1:i + 1 + h]
            level = float(c["breakout_level"])
            row[f"hit_{h}d"] = bool(float(future["High"].max()) >= level)
            row[f"return_{h}d_pct"] = (float(future["Close"].iloc[-1]) / entry - 1) * 100
            row[f"max_{h}d_pct"] = (float(future["High"].max()) / entry - 1) * 100
            row[f"dd_{h}d_pct"] = (float(future["Low"].min()) / entry - 1) * 100
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(events, hold_days):
    if events.empty:
        return pd.DataFrame()
    out = []
    for rating in range(1, 11):
        x = events[events.rating == rating]
        if x.empty:
            continue
        r = {"rating": rating, "signals": len(x), "false_breakout_pct": x.false_breakout.mean() * 100}
        for h in hold_days:
            r[f"hit_{h}d_pct"] = x[f"hit_{h}d"].mean() * 100
            r[f"avg_return_{h}d_pct"] = x[f"return_{h}d_pct"].mean()
            r[f"median_return_{h}d_pct"] = x[f"return_{h}d_pct"].median()
            r[f"avg_max_{h}d_pct"] = x[f"max_{h}d_pct"].mean()
            r[f"avg_drawdown_{h}d_pct"] = x[f"dd_{h}d_pct"].mean()
        out.append(r)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--hold-days", default="5,10,20")
    ap.add_argument("--out", default="validation_output")
    args = ap.parse_args()
    hold_days = [int(x) for x in args.hold_days.split(",")]

    universe = scanner.load_universe()
    symbols = universe.symbol.tolist()
    print(f"NIFTY 200 universe rows: {len(symbols)}")
    histories = download_batch(symbols, args.years)
    print(f"Usable price histories: {len(histories)}")

    events = []
    for n, (symbol, df) in enumerate(histories.items(), 1):
        x = validate_stock(symbol, df, hold_days)
        if not x.empty:
            events.append(x)
        if n % 20 == 0:
            print(f"Processed {n}/{len(histories)} stocks")
    events = pd.concat(events, ignore_index=True) if events else pd.DataFrame()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    events.to_csv(out / "validation_events.csv", index=False)
    summary = summarize(events, hold_days)
    summary.to_csv(out / "score_summary.csv", index=False)

    # Simple chronological out-of-sample split: first 70% vs last 30% of signal dates.
    if not events.empty:
        cutoff = events.date.quantile(0.70)
        split = pd.DataFrame({
            "period": ["train_70pct", "out_of_sample_30pct"],
            "from": [events.date.min(), cutoff],
            "to": [cutoff, events.date.max()],
            "signals": [int((events.date <= cutoff).sum()), int((events.date > cutoff).sum())],
        })
        split.to_csv(out / "time_split.csv", index=False)

    print(f"Validation events: {len(events)}")
    print(summary.to_string(index=False) if not summary.empty else "No validation events generated.")
    print(f"Reports written to: {out.resolve()}")


if __name__ == "__main__":
    main()
