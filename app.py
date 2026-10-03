import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import universe

# --- PAGE CONFIGURATION & MINIMALIST STYLING ---
st.set_page_config(
    page_title="Nifty 100 Breakout & Pattern Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Minimalist CSS Styling
st.markdown("""
<style>
    .reportview-container {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #161b22;
        border-radius: 8px;
        padding: 14px 18px;
        border: 1px solid #30363d;
        margin-bottom: 12px;
    }
    .badge-breakout {
        background-color: rgba(35, 134, 54, 0.2);
        color: #3fb950;
        border: 1px solid #238636;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-recent {
        background-color: rgba(56, 139, 253, 0.2);
        color: #58a6ff;
        border: 1px solid #1f6feb;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-coiling {
        background-color: rgba(210, 153, 34, 0.2);
        color: #d29922;
        border: 1px solid #9e6a03;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)


# --- CANDLESTICK PATTERN & MULTI-POINT DETECTION ENGINE ---
class TechnicalEngine:
    @staticmethod
    def detect_candlestick_patterns(df: pd.DataFrame) -> list:
        """
        Detects prominent bullish reversal and continuation candlestick patterns on the latest candles.
        """
        if len(df) < 5:
            return []
        
        patterns = []
        c = df['Close'].values
        o = df['Open'].values
        h = df['High'].values
        l = df['Low'].values
        
        # Last candle (-1) and previous (-2, -3)
        body_now = abs(c[-1] - o[-1])
        range_now = h[-1] - l[-1] if (h[-1] - l[-1]) > 0 else 0.001
        is_bullish_now = c[-1] > o[-1]
        
        body_prev = abs(c[-2] - o[-2])
        is_bearish_prev = c[-2] < o[-2]
        
        # 1. Bullish Marubozu (Strong directional buyer conviction)
        if is_bullish_now and (body_now / range_now > 0.85):
            patterns.append("Bullish Marubozu (High Conviction)")

        # 2. Bullish Engulfing
        if is_bearish_prev and is_bullish_now:
            if c[-1] > o[-2] and o[-1] < c[-2]:
                patterns.append("Bullish Engulfing")

        # 3. Hammer / Pin Bar (Rejection of lower prices)
        lower_shadow = min(c[-1], o[-1]) - l[-1]
        upper_shadow = h[-1] - max(c[-1], o[-1])
        if lower_shadow >= 2 * body_now and upper_shadow <= 0.2 * body_now:
            patterns.append("Bullish Hammer / Pin Bar")

        # 4. Inside Bar (Volatility compression before explosion)
        if h[-1] < h[-2] and l[-1] > l[-2]:
            patterns.append("Inside Bar (Range Contraction)")
            
        # 5. Morning Star pattern (3 candles)
        if len(df) >= 3:
            is_bearish_p2 = c[-3] < o[-3]
            small_body_p1 = abs(c[-2] - o[-2]) < (0.35 * abs(c[-3] - o[-3]))
            if is_bearish_p2 and small_body_p1 and is_bullish_now and (c[-1] > (c[-3] + o[-3]) / 2):
                patterns.append("Morning Star Reversal")

        return patterns if patterns else ["Consolidation Candle"]

    @staticmethod
    def analyze_stock(df: pd.DataFrame, bench_df: pd.DataFrame) -> dict:
        """
        Multi-point verification:
        1. 50-day rolling resistance calculation
        2. Proximity / Breakout status check (Live vs Last 5 Days)
        3. Trend alignment (Price > 20 EMA > 50 EMA > 200 SMA)
        4. Bollinger Band Squeeze (Volatility Contraction)
        5. Relative Strength (RS) Mansfield-style benchmark score
        6. Volume Spike vs 20 SMA
        7. Candlestick Patterns
        """
        if len(df) < 55:
            return None

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        current_price = close.iloc[-1]
        prev_close = close.iloc[-2]
        current_volume = volume.iloc[-1]

        # Multi-timeframe trend alignment
        ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        sma_200 = close.rolling(window=min(200, len(close))).mean().iloc[-1]
        trend_score = 0
        if current_price > sma_200: trend_score += 1
        if current_price > ema_20: trend_score += 1
        if ema_20 > ema_50: trend_score += 1

        # Resistance calculation (50-day high excluding today)
        resistance_50d = high.iloc[-51:-1].max()
        dist_pct = ((resistance_50d - current_price) / resistance_50d) * 100

        # Check for Breakout in the Last 1 Week (last 5 trading sessions)
        crossed_in_last_week = False
        breakout_day_str = None
        for i in range(2, min(7, len(df))):
            past_close = close.iloc[-i]
            prior_res = high.iloc[-(50 + i):-i].max()
            if past_close >= prior_res:
                crossed_in_last_week = True
                breakout_day_str = df.index[-i].strftime('%d %b')
                break

        # Volatility squeeze check
        rolling_mean = close.rolling(window=20).mean()
        rolling_std = close.rolling(window=20).std()
        upper_bb = rolling_mean + (2 * rolling_std)
        lower_bb = rolling_mean - (2 * rolling_std)
        bb_width = (upper_bb - lower_bb) / rolling_mean
        is_squeezing = bb_width.iloc[-1] < bb_width.rolling(window=40).mean().iloc[-1]

        # Volume multiple
        avg_vol_20 = volume.rolling(window=20).mean().iloc[-1]
        vol_multiple = current_volume / avg_vol_20 if avg_vol_20 > 0 else 0

        # Relative Strength vs Nifty 50 over 20 days
        stock_perf_20 = (current_price - close.iloc[-21]) / close.iloc[-21]
        bench_close = bench_df['Close']
        bench_perf_20 = (bench_close.iloc[-1] - bench_close.iloc[-21]) / bench_close.iloc[-21] if len(bench_close) >= 21 else 0
        rs_score = round((stock_perf_20 - bench_perf_20) * 100, 2)

        # Candlestick pattern detection
        candlestick_patterns = TechnicalEngine.detect_candlestick_patterns(df)

        # Status Classification
        if current_price >= resistance_50d and vol_multiple >= 1.2:
            status = "Breakout Confirmed"
            badge = "badge-breakout"
            category = "Live Breakouts"
        elif crossed_in_last_week and current_price >= (resistance_50d * 0.98):
            status = f"Broke Out ({breakout_day_str})"
            badge = "badge-recent"
            category = "Breakout in Last 1 Week"
        elif 0.0 <= dist_pct <= 3.5:
            status = "Coiling Near Breakout"
            badge = "badge-coiling"
            category = "Near Breakout (Watchlist)"
        else:
            category = "Other"
            status = "In Range"
            badge = "badge-coiling"

        return {
            "current_price": round(current_price, 2),
            "pct_change": round(((current_price - prev_close) / prev_close) * 100, 2),
            "resistance_50d": round(resistance_50d, 2),
            "dist_resistance_pct": round(dist_pct, 2),
            "vol_multiple": round(vol_multiple, 2),
            "rs_score": rs_score,
            "trend_score": f"{trend_score}/3",
            "is_squeezing": "Yes" if is_squeezing else "No",
            "status": status,
            "badge": badge,
            "category": category,
            "candlestick_patterns": ", ".join(candlestick_patterns)
        }


# --- CACHED DATA FETCHING ---
@st.cache_data(ttl=300)
def fetch_data(tickers: list):
    batch = yf.download(tickers=tickers, period="1y", interval="1d", group_by='ticker', auto_adjust=False, threads=True)
    bench = yf.download(universe.INDEX_TICKER, period="1y", interval="1d", progress=False)
    return batch, bench


# --- BACKTESTING ENGINE FOR STRATEGY OPTIMIZATION ---
def run_backtest(tickers: list, hold_days=10, profit_target=0.08, stop_loss=0.04):
    """
    Backtests the breakout logic across the universe over the past 1 year.
    Evaluates:
    - Win Rate (%)
    - Average Return per Trade (%)
    - Max Profit Trade / Max Drawdown Trade
    """
    trades = []
    
    with st.spinner("Backtesting breakout logic over past 12 months..."):
        batch, _ = fetch_data(tickers)
        for ticker in tickers:
            try:
                df = batch[ticker].dropna() if len(tickers) > 1 else batch.dropna()
                if len(df) < 120:
                    continue
                
                close = df['Close'].values
                high = df['High'].values
                low = df['Low'].values
                vol = df['Volume'].values
                dates = df.index
                
                # Iterate from day 60 up to (total - hold_days)
                for i in range(60, len(df) - hold_days):
                    # 50-day resistance prior to day i
                    past_res = high[i-50:i].max()
                    avg_vol = vol[i-20:i].mean()
                    
                    # Condition: Decisive Breakout with Volume Surge
                    if close[i] > past_res and vol[i] > 1.3 * avg_vol:
                        entry_price = close[i]
                        exit_price = close[i + hold_days]
                        
                        # Track intra-trade target / stop loss hits
                        outcome = "Hold Complete"
                        for d in range(1, hold_days + 1):
                            idx = i + d
                            if high[idx] >= entry_price * (1 + profit_target):
                                exit_price = entry_price * (1 + profit_target)
                                outcome = "Target Hit"
                                break
                            elif low[idx] <= entry_price * (1 - stop_loss):
                                exit_price = entry_price * (1 - stop_loss)
                                outcome = "Stop Hit"
                                break
                                
                        pct_ret = round(((exit_price - entry_price) / entry_price) * 100, 2)
                        trades.append({
                            "Ticker": ticker.replace(".NS", ""),
                            "Date": dates[i].strftime("%Y-%m-%d"),
                            "Entry": round(entry_price, 2),
                            "Exit": round(exit_price, 2),
                            "Return (%)": pct_ret,
                            "Outcome": outcome
                        })
            except Exception:
                continue

    if not trades:
        return pd.DataFrame(), {}

    trades_df = pd.DataFrame(trades)
    win_trades = trades_df[trades_df['Return (%)'] > 0]
    win_rate = round((len(win_trades) / len(trades_df)) * 100, 2)
    avg_return = round(trades_df['Return (%)'].mean(), 2)
    profit_factor = round(abs(win_trades['Return (%)'].sum() / (trades_df[trades_df['Return (%)'] <= 0]['Return (%)'].sum() or 0.001)), 2)

    stats = {
        "Total Signals": len(trades_df),
        "Win Rate (%)": f"{win_rate}%",
        "Average Return": f"{avg_return}%",
        "Profit Factor": profit_factor,
        "Best Trade": f"+{trades_df['Return (%)'].max()}%",
        "Worst Trade": f"{trades_df['Return (%)'].min()}%"
    }
    return trades_df, stats


# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("### 🎛️ Terminal Navigation")
mode = st.sidebar.radio("Select View Mode", ["Live Scanner & Patterns", "Strategy Backtesting"])

st.sidebar.markdown("---")
if mode == "Live Scanner & Patterns":
    st.sidebar.subheader("Scanner Sensitivity")
    max_dist = st.sidebar.slider("Max Distance to Resistance (%)", 0.5, 5.0, 3.5, 0.5)
    min_vol = st.sidebar.slider("Min Volume Surge vs 20 SMA", 0.8, 3.0, 1.1, 0.1)
    require_full_trend = st.sidebar.checkbox("Strict Trend Filter (Price > 20 EMA > 50 EMA > 200 SMA)", value=False)
    show_only = st.sidebar.selectbox("Filter by Category", ["All Categories", "Live Breakouts", "Breakout in Last 1 Week", "Near Breakout (Watchlist)"])
else:
    st.sidebar.subheader("Backtest Parameters")
    holding_period = st.sidebar.slider("Holding Horizon (Trading Days)", 3, 25, 10)
    tp = st.sidebar.slider("Profit Target (%)", 4, 20, 8) / 100.0
    sl = st.sidebar.slider("Stop Loss (%)", 2, 10, 4) / 100.0


# --- VIEW 1: LIVE SCANNER & CANDLESTICK PATTERNS ---
if mode == "Live Scanner & Patterns":
    st.title("⚡ Nifty 100 Breakout Terminal")
    st.caption("Live institutional accumulation, multi-point trend verification, and candlestick pattern intelligence.")

    if st.button("🔍 Scan Nifty 100 Now", use_container_width=True):
        batch, bench = fetch_data(universe.NIFTY_100)
        results = []
        for t in universe.NIFTY_100:
            try:
                sdf = batch[t].dropna() if len(universe.NIFTY_100) > 1 else batch.dropna()
                metrics = TechnicalEngine.analyze_stock(sdf, bench)
                if not metrics:
                    continue
                
                # Check user criteria
                passes_dist = metrics['dist_resistance_pct'] <= max_dist or metrics['category'] in ["Live Breakouts", "Breakout in Last 1 Week"]
                passes_vol = metrics['vol_multiple'] >= min_vol or metrics['category'] == "Breakout in Last 1 Week"
                passes_trend = (metrics['trend_score'] == "3/3") if require_full_trend else True
                
                if passes_dist and passes_vol and passes_trend:
                    if metrics['category'] != "Other":
                        results.append({
                            "Ticker": t.replace(".NS", ""),
                            "FullTicker": t,
                            "Status": metrics['status'],
                            "Category": metrics['category'],
                            "LTP (₹)": metrics['current_price'],
                            "Day Change (%)": metrics['pct_change'],
                            "50D Resistance (₹)": metrics['resistance_50d'],
                            "Dist to Breakout (%)": metrics['dist_resistance_pct'],
                            "Vol Surge": f"{metrics['vol_multiple']}x",
                            "RS vs Nifty": metrics['rs_score'],
                            "Trend Score": metrics['trend_score'],
                            "Candlestick Patterns": metrics['candlestick_patterns']
                        })
            except Exception:
                continue

        st.session_state['scan_data'] = pd.DataFrame(results)
        st.session_state['batch_data'] = batch

    if 'scan_data' in st.session_state and not st.session_state['scan_data'].empty:
        df = st.session_state['scan_data']
        
        # Apply Category Filter if selected
        if show_only != "All Categories":
            df = df[df['Category'] == show_only]

        # Top Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Setups Identified", len(df))
        c2.metric("Live Breakouts (Today)", len(df[df['Category'] == "Live Breakouts"]))
        c3.metric("Broke Out Last 1 Week", len(df[df['Category'] == "Breakout in Last 1 Week"]))
        c4.metric("Coiling Near Ceiling", len(df[df['Category'] == "Near Breakout (Watchlist)"]))

        st.markdown("---")
        st.subheader("📋 Screened Candidates")
        
        # Display Table with interactive columns
        st.dataframe(
            df[['Ticker', 'Category', 'Status', 'LTP (₹)', 'Day Change (%)', '50D Resistance (₹)', 'Dist to Breakout (%)', 'Vol Surge', 'Candlestick Patterns', 'RS vs Nifty', 'Trend Score']],
            column_config={
                "LTP (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                "50D Resistance (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                "Day Change (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "Dist to Breakout (%)": st.column_config.NumberColumn(format="%.2f%%"),
            },
            use_container_width=True,
            hide_index=True
        )

        # Detailed Chart & Candlestick Pattern Inspection
        st.markdown("---")
        st.subheader("📊 Stock Chart Pattern & Candlestick Analyzer")
        selected_stock = st.selectbox("Select a stock to inspect detailed setup & candlestick action:", df['Ticker'].tolist())
        
        if selected_stock:
            row = df[df['Ticker'] == selected_stock].iloc[0]
            st_full = row['FullTicker']
            sdf = st.session_state['batch_data'][st_full].dropna().tail(130)
            
            # Stock Metrics banner
            st.markdown(f"""
            **Stock:** `{selected_stock}` | **Category:** `{row['Category']}` | **Status:** `{row['Status']}`  
            **Identified Candlestick Signal:** <span style="color:#3fb950; font-weight:600;">{row['Candlestick Patterns']}</span> | **Trend Alignment:** `{row['Trend Score']}`
            """, unsafe_allow_html=True)
            
            # Interactive Candlestick Chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=sdf.index,
                open=sdf['Open'],
                high=sdf['High'],
                low=sdf['Low'],
                close=sdf['Close'],
                name="Candlesticks"
            ))
            
            # Moving Averages
            ema20 = sdf['Close'].ewm(span=20, adjust=False).mean()
            ema50 = sdf['Close'].ewm(span=50, adjust=False).mean()
            sma200 = sdf['Close'].rolling(window=min(200, len(sdf))).mean()
            fig.add_trace(go.Scatter(x=sdf.index, y=ema20, line=dict(color='#ff9900', width=1.2), name="20 EMA"))
            fig.add_trace(go.Scatter(x=sdf.index, y=ema50, line=dict(color='#00b4d8', width=1.2), name="50 EMA"))
            
            # 50D Resistance Line
            res_val = row['50D Resistance (₹)']
            fig.add_hline(
                y=res_val,
                line_dash="dash",
                line_color="#ff4b4b",
                annotation_text=f"Breakout Ceiling: ₹{res_val}",
                annotation_position="top right"
            )
            
            fig.update_layout(
                title=f"{selected_stock} — Technical Breakout Structure",
                yaxis_title="Price (₹)",
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                height=520,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)


# --- VIEW 2: STRATEGY BACKTESTING ENGINE ---
else:
    st.title("🧪 Quantitative Backtesting & Verification Engine")
    st.caption("Validates the statistical edge of this pre-breakout strategy on Nifty 100 historical data.")
    
    st.info(f"Target: +{int(tp*100)}% | Stop Loss: -{int(sl*100)}% | Holding Period: Max {holding_period} days")
    
    if st.button("🚀 Run Backtest Simulation", use_container_width=True):
        trades_df, stats = run_backtest(
            tickers=universe.NIFTY_100,
            hold_days=holding_period,
            profit_target=tp,
            stop_loss=sl
        )
        
        if not trades_df.empty:
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Win Rate", stats["Win Rate (%)"])
            b2.metric("Average Trade Return", stats["Average Return"])
            b3.metric("Profit Factor", stats["Profit Factor"])
            b4.metric("Total Trade Signals", stats["Total Signals"])
            
            st.markdown("---")
            st.subheader("📜 Historical Trade Log")
            st.dataframe(trades_df.tail(100), use_container_width=True, hide_index=True)
        else:
            st.warning("No historical trades met the simulation threshold in this window.")
