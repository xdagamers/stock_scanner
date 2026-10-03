import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import universe

# --- PAGE CONFIGURATION & NEO-BRUTALIST STYLING ---
st.set_page_config(
    page_title="NeoBreakout Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mobile-Optimized Neo-Brutalist UI CSS
st.markdown("""
<style>
    :root {
        --bg-color: #f4f4f0;
        --card-bg: #ffffff;
        --accent-green: #a3e635;
        --accent-blue: #60a5fa;
        --accent-yellow: #fde047;
        --text-dark: #111827;
        --border-dark: #000000;
        --shadow-brutal: 4px 4px 0px #000000;
    }
    
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-dark);
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* Hide standard st metric cards */
    [data-testid="stMetricValue"] { display: none; }
    
    /* Neo-Brutalist Grid Container */
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    /* Neo-Brutalist Card */
    .brutal-card {
        background-color: var(--card-bg);
        border: 3px solid var(--border-dark);
        box-shadow: var(--shadow-brutal);
        border-radius: 8px;
        padding: 1.5rem;
        transition: transform 0.1s;
    }
    .brutal-card:active {
        transform: translate(2px, 2px);
        box-shadow: 2px 2px 0px #000000;
    }
    
    .card-title {
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .card-value {
        font-size: 2.2rem;
        font-weight: 900;
    }
    
    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border: 2px solid var(--border-dark);
        box-shadow: 2px 2px 0px #000;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
    }
    .bg-green { background-color: var(--accent-green); }
    .bg-blue { background-color: var(--accent-blue); }
    .bg-yellow { background-color: var(--accent-yellow); }
    
    /* Custom Streamlit Dataframe Styling */
    [data-testid="stDataFrame"] {
        border: 3px solid var(--border-dark);
        box-shadow: var(--shadow-brutal);
        border-radius: 8px;
        background-color: white;
    }
    
    /* Streamlit Sidebar overrides */
    [data-testid="stSidebar"] {
        background-color: white;
        border-right: 3px solid var(--border-dark);
    }
</style>
""", unsafe_allow_html=True)


# --- CANDLESTICK PATTERN ENGINE ---
class TechnicalEngine:
    @staticmethod
    def detect_candlestick_patterns(df: pd.DataFrame) -> list:
        if len(df) < 5: return []
        patterns = []
        c, o, h, l = df['Close'].values, df['Open'].values, df['High'].values, df['Low'].values
        
        body_now = abs(c[-1] - o[-1])
        range_now = h[-1] - l[-1] if (h[-1] - l[-1]) > 0 else 0.001
        is_bullish_now = c[-1] > o[-1]
        body_prev = abs(c[-2] - o[-2])
        is_bearish_prev = c[-2] < o[-2]
        
        if is_bullish_now and (body_now / range_now > 0.85): patterns.append("Bullish Marubozu")
        if is_bearish_prev and is_bullish_now and c[-1] > o[-2] and o[-1] < c[-2]: patterns.append("Bullish Engulfing")
        
        lower_shadow = min(c[-1], o[-1]) - l[-1]
        upper_shadow = h[-1] - max(c[-1], o[-1])
        if lower_shadow >= 2 * body_now and upper_shadow <= 0.2 * body_now: patterns.append("Hammer Pin")
        
        if h[-1] < h[-2] and l[-1] > l[-2]: patterns.append("Inside Bar Squeeze")
        
        if len(df) >= 3:
            if c[-3] < o[-3] and abs(c[-2] - o[-2]) < (0.35 * abs(c[-3] - o[-3])) and is_bullish_now and c[-1] > (c[-3] + o[-3]) / 2:
                patterns.append("Morning Star")

        return patterns if patterns else ["Consolidation"]

    @staticmethod
    def analyze_stock(df: pd.DataFrame, bench_df: pd.DataFrame) -> dict:
        if len(df) < 55: return None
        
        close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']
        current_price, prev_close = close.iloc[-1], close.iloc[-2]
        current_volume = volume.iloc[-1]
        avg_vol_20 = volume.rolling(window=20).mean().iloc[-1]
        vol_multiple = current_volume / avg_vol_20 if avg_vol_20 > 0 else 0

        sma_200 = close.rolling(window=min(200, len(close))).mean().iloc[-1]
        ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]

        # Resistance calculation (50-day high excluding today)
        resistance_50d = high.iloc[-51:-1].max()
        dist_pct = ((resistance_50d - current_price) / resistance_50d) * 100

        # Breakout in last week (last 5 sessions)
        crossed_in_last_week = False
        breakout_day_str = ""
        for i in range(2, min(7, len(df))):
            if close.iloc[-i] >= high.iloc[-(50 + i):-i].max():
                crossed_in_last_week = True
                breakout_day_str = f"({df.index[-i].strftime('%d %b')})"
                break

        # Categorization Logic (Correcting the volume exclusion bug)
        is_live_breakout = (current_price >= resistance_50d) and (vol_multiple >= 1.2)
        is_recent_breakout = crossed_in_last_week and (current_price >= (resistance_50d * 0.98))
        is_coiling_watchlist = (0.0 <= dist_pct <= 4.0) and not is_live_breakout
        
        if is_live_breakout:
            category = "🚀 Live Breakouts"
            status = "Breakout Confirmed"
        elif is_recent_breakout:
            category = "📅 Broke Out Last Week"
            status = f"Broke Out {breakout_day_str}"
        elif is_coiling_watchlist:
            category = "👁️ Watchlist (Coiling)"
            status = "Squeezing near resistance"
        else:
            return None # Skip stocks far from breakout or broken trends

        stock_perf = (current_price - close.iloc[-21]) / close.iloc[-21]
        bench_perf = (bench_df['Close'].iloc[-1] - bench_df['Close'].iloc[-21]) / bench_df['Close'].iloc[-21]
        rs_score = round((stock_perf - bench_perf) * 100, 2)
        
        trend_score = sum([current_price > sma_200, current_price > ema_20, ema_20 > ema_50])
        
        return {
            "current_price": round(current_price, 2),
            "pct_change": round(((current_price - prev_close) / prev_close) * 100, 2),
            "resistance_50d": round(resistance_50d, 2),
            "dist_resistance_pct": round(dist_pct, 2),
            "vol_multiple": round(vol_multiple, 2),
            "rs_score": rs_score,
            "trend_score": f"{trend_score}/3",
            "status": status,
            "category": category,
            "candlestick_patterns": ", ".join(TechnicalEngine.detect_candlestick_patterns(df))
        }

@st.cache_data(ttl=300)
def fetch_data(tickers: list):
    batch = yf.download(tickers=tickers, period="1y", interval="1d", group_by='ticker', auto_adjust=False, threads=True)
    bench = yf.download(universe.INDEX_TICKER, period="1y", interval="1d", progress=False)
    return batch, bench

# --- SIDEBAR & HEADER ---
st.sidebar.markdown("<h2 style='font-weight:900;'>⚙️ TERMINAL SETTINGS</h2>", unsafe_allow_html=True)
st.sidebar.info("Universe: Top ~125 NSE Stocks (Mkt Cap > ₹50,000 Cr)")

st.markdown("<h1 style='font-weight: 900; letter-spacing: -1px;'>⚡ NEO-BREAKOUT SCANNER</h1>", unsafe_allow_html=True)
st.markdown("<b>Live quantitative tracking for large-cap institutional breakouts & volatility contractions.</b>", unsafe_allow_html=True)
st.write("")

# --- MAIN EXECUTION ---
if st.button("🚀 INITIATE MARKET SCAN", use_container_width=True, type="primary"):
    with st.spinner("Downloading 1Y data for 125 large-caps and running quantitative arrays..."):
        batch, bench = fetch_data(universe.LARGE_CAP_50K_CR)
        results = []
        for t in universe.LARGE_CAP_50K_CR:
            try:
                sdf = batch[t].dropna() if len(universe.LARGE_CAP_50K_CR) > 1 else batch.dropna()
                metrics = TechnicalEngine.analyze_stock(sdf, bench)
                if metrics:
                    results.append({
                        "Ticker": t.replace(".NS", ""),
                        "FullTicker": t,
                        "Category": metrics['category'],
                        "Status": metrics['status'],
                        "LTP (₹)": metrics['current_price'],
                        "Day Chg (%)": metrics['pct_change'],
                        "50D High (₹)": metrics['resistance_50d'],
                        "Dist to BRK (%)": metrics['dist_resistance_pct'],
                        "Vol Surge": f"{metrics['vol_multiple']}x",
                        "Vol Sort": metrics['vol_multiple'], # Hidden column for sorting
                        "Pattern": metrics['candlestick_patterns'],
                        "RS vs Nifty": metrics['rs_score']
                    })
            except Exception:
                continue

        st.session_state['scan_data'] = pd.DataFrame(results)
        st.session_state['batch_data'] = batch

if 'scan_data' in st.session_state and not st.session_state['scan_data'].empty:
    df = st.session_state['scan_data']
    
    cnt_live = len(df[df['Category'] == "🚀 Live Breakouts"])
    cnt_recent = len(df[df['Category'] == "📅 Broke Out Last Week"])
    cnt_watch = len(df[df['Category'] == "👁️ Watchlist (Coiling)"])

    # Neo-Brutalist Metric Grid
    st.markdown(f"""
    <div class="grid-container">
        <div class="brutal-card">
            <div class="card-title">Live Breakouts</div>
            <div class="card-value"><span class="badge bg-green">{cnt_live}</span></div>
        </div>
        <div class="brutal-card">
            <div class="card-title">Broke Out Last 1 Week</div>
            <div class="card-value"><span class="badge bg-blue">{cnt_recent}</span></div>
        </div>
        <div class="brutal-card">
            <div class="card-title">Pre-Breakout Watchlist</div>
            <div class="card-value"><span class="badge bg-yellow">{cnt_watch}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 SCANNER RESULTS (Ranked by Momentum)")
    
    # Display separate dataframes for each category, sorted optimally
    tabs = st.tabs(["🚀 Live Breakouts", "📅 Last 1 Week", "👁️ Pre-Breakout Watchlist"])
    
    def display_category_table(category_name, sort_col, asc=False):
        sub_df = df[df['Category'] == category_name]
        if sub_df.empty:
            st.info(f"No stocks currently matching {category_name}.")
            return
        
        # Sort data descending by Day Change or Volume Surge
        sub_df = sub_df.sort_values(by=sort_col, ascending=asc)
        
        st.dataframe(
            sub_df.drop(columns=['FullTicker', 'Category', 'Vol Sort']),
            column_config={
                "LTP (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                "50D High (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                "Day Chg (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "Dist to BRK (%)": st.column_config.NumberColumn(format="%.2f%%"),
            },
            use_container_width=True,
            hide_index=True
        )

    with tabs[0]: display_category_table("🚀 Live Breakouts", "Day Chg (%)", False)
    with tabs[1]: display_category_table("📅 Broke Out Last Week", "Day Chg (%)", False)
    with tabs[2]: display_category_table("👁️ Watchlist (Coiling)", "Dist to BRK (%)", True) # Sort watchlist by closest distance first

    # Interactive Charting Feature
    st.markdown("<br><hr style='border:1px solid #000;'><br>", unsafe_allow_html=True)
    st.markdown("### 📈 CHART & CANDLESTICK ANALYZER")
    selected_stock = st.selectbox("Select a scanned stock to inspect price action:", df['Ticker'].tolist())
    
    if selected_stock:
        row = df[df['Ticker'] == selected_stock].iloc[0]
        sdf = st.session_state['batch_data'][row['FullTicker']].dropna().tail(120)
        
        st.markdown(f"**Status:** `{row['Status']}` | **Detected Pattern:** `{row['Pattern']}` | **RS Score:** `{row['RS vs Nifty']}`")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=sdf.index, open=sdf['Open'], high=sdf['High'], low=sdf['Low'], close=sdf['Close'], name="Price"
        ))
        
        # Plot Averages & Resistance
        fig.add_trace(go.Scatter(x=sdf.index, y=sdf['Close'].ewm(span=20).mean(), line=dict(color='orange', width=2), name="20 EMA"))
        fig.add_hline(y=row['50D High (₹)'], line_dash="solid", line_color="black", line_width=2, annotation_text="Breakout Ceiling")
        
        fig.update_layout(
            title=f"{selected_stock} Daily Structure",
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            plot_bgcolor="#f4f4f0",
            paper_bgcolor="#f4f4f0",
            font=dict(color="#000", family="Inter"),
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
