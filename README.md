# NIFTY 200 Breakout Scanner

A Streamlit-ready Indian stock technical scanner for the NIFTY 200.

## What it does

- Loads the NIFTY 200 constituent universe.
- Downloads daily EOD OHLCV data.
- Finds a nearby resistance / breakout level from recent swing highs.
- Calculates:
  - Breakout distance
  - 1–10 setup rating
  - Resistance quality
  - Relative volume (RVOL)
  - RSI
  - MACD
  - ADX / DMI
  - 20/50/200 moving-average structure
  - Candlestick pattern
  - 60-day price performance
- Watchlist order:
  1. nearest to breakout
  2. setup score
  3. volume confirmation
- Includes search and an interactive candlestick chart.
- Allows CSV download of the watchlist.
- Walk-forward historical backtesting with breakout-hit rate and return statistics.
- Recent false-breakout detection and scoring penalty.
- NIFTY relative-strength comparison.
- Zone-based resistance detection using clustered swing highs.

## Data source

The app first requests the NIFTY 200 constituent CSV from NSE's official archive endpoint. If that request is unavailable in the hosting environment, it uses a fallback mirror.

Historical daily market data is retrieved through `yfinance` from Yahoo Finance.

This is an EOD/research prototype. It is **not a real-time market-data feed** and should not be treated as a guaranteed trading signal.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload all files from this folder.
3. Go to Streamlit Community Cloud.
4. Choose **Create app**.
5. Select your GitHub repository.
6. Select `streamlit_app.py` as the entrypoint.
7. Deploy.

`requirements.txt` is already included.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Important limitations

- Yahoo Finance/yfinance availability and rate limits can change.
- NSE constituent membership changes over time; the app refreshes the list when it runs.
- Daily data means breakout confirmation is based on daily bars, not tick-by-tick intraday prices.
- A 10/10 score is a technical setup score, not a prediction, probability, or buy/sell recommendation.
- Before using this for real money, backtest the rules on historical data with transaction costs, slippage and survivorship-bias controls.

## Algorithm notes

### Resistance
Resistance is treated as a zone rather than a single arbitrary high. Recent swing highs are clustered within a tolerance band, then the nearest meaningful zone is selected using distance, touch count and recency.

### False breakouts
A recent move above prior resistance followed by a close back below that level within a few bars is flagged. The scanner applies a score penalty rather than treating the event as a bullish breakout.

### NIFTY strength
The scanner compares the stock's 60-day return with NIFTY 50 to measure relative strength.

### Backtesting
The backtest walks forward one historical bar at a time. It never uses future bars to create the signal. Future bars are used only for outcome measurement.

## Scoring calibration

The scanner now separates each indicator into a normalized component score and applies configurable weights. The default weights remain conservative:

- Proximity 20
- Resistance 15
- Volume 15
- RSI 10
- MACD 10
- ADX 10
- Moving averages 10
- Candlestick 5
- Relative strength 5

A built-in tuning tool tests a small set of alternative weight configurations over historical data and reports breakout hit rate and forward returns. It does **not** automatically replace production weights, which helps reduce accidental overfitting.

## Next development phase

The architecture is intentionally simple so the following can be added without rebuilding the UI:

- historical backtesting
- NIFTY/sector relative strength
- fundamentals
- breakout confirmation after a daily close
- false-breakout detection
- alerts
- broker API / licensed real-time feed

## Test status

The package includes `test_scanner.py` with synthetic-data tests for indicators, score bounds, walk-forward backtesting, and weight optimization. The scanner module has been syntax-checked. Live Yahoo/NSE execution depends on network access and installed dependencies in the hosting environment.

The weight optimizer was optimized to calculate historical feature rows once and reuse them across candidate weight sets.
