import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from scanner import scan_universe, scan_single, load_universe, fetch_history

st.set_page_config(
    page_title="NIFTY 200 Breakout Scanner",
    page_icon="📈",
    layout="wide",
)

st.title("📈 NIFTY 200 Breakout Scanner")
st.caption(
    "Daily EOD technical scanner • 1–10 setup score • nearest breakout first • "
    "Educational/research tool, not a buy/sell recommendation."
)

with st.sidebar:
    st.header("Scanner Settings")
    max_distance = st.slider("Maximum breakout distance (%)", 2.0, 15.0, 10.0, 0.5)
    min_score = st.slider("Minimum setup score", 1, 10, 5)
    period = st.selectbox("History", ["1y", "2y"], index=1)
    st.markdown("---")
    st.write("**Data:** Yahoo Finance EOD via yfinance")
    st.write("**Universe:** NIFTY 200")
    st.write("**Refresh:** click Scan / Refresh data")

@st.cache_data(ttl=3600, show_spinner=False)
def cached_scan(max_distance, period):
    return scan_universe(max_distance=max_distance, period=period)

universe = load_universe()

tab1, tab2, tab3 = st.tabs(["🔎 NIFTY 200 Scanner", "📊 Search & Chart", "🧪 Backtest"])

with tab1:
    st.subheader(f"Watchlist ({len(universe)} NIFTY 200 constituents)")
    st.info(
        "Ranking order: nearest to breakout → setup score → volume confirmation. "
        "Only stocks within your selected distance are shown."
    )

    if st.button("🚀 Scan / Refresh", type="primary", use_container_width=True):
        st.cache_data.clear()
        with st.spinner("Scanning NIFTY 200. This can take a little while because historical data is downloaded..."):
            result = scan_universe(max_distance=max_distance, period=period)
        st.session_state["scan_result"] = result

    result = st.session_state.get("scan_result")
    if result is None:
        st.warning("Click **Scan / Refresh** to run the scanner.")
    else:
        if result.empty:
            st.warning("No stocks matched the selected distance/score filters.")
        else:
            shown = result[result["rating"] >= min_score].copy()
            st.metric("Stocks found", len(shown))

            display_cols = [
                "symbol", "price", "breakout_level", "distance_pct",
                "rating", "status", "rsi", "macd", "adx", "rvol",
                "ema_structure", "candlestick", "relative_strength", "nifty_relative_strength", "sector_relative_strength",
                "false_breakout"
            ]
            display_cols = [c for c in display_cols if c in shown.columns]

            st.dataframe(
                shown[display_cols].style.format({
                    "price": "₹{:.2f}",
                    "breakout_level": "₹{:.2f}",
                    "distance_pct": "{:.2f}%",
                    "rsi": "{:.1f}",
                    "macd": "{:.2f}",
                    "adx": "{:.1f}",
                    "rvol": "{:.2f}x",
                    "relative_strength": "{:.1f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )

            csv = shown.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download watchlist CSV",
                csv,
                "nifty200_breakout_watchlist.csv",
                "text/csv",
            )

with tab2:
    st.subheader("Search a NIFTY 200 stock")
    symbols = universe["symbol"].dropna().astype(str).tolist()
    query = st.text_input("Search by symbol or company name", placeholder="e.g. KOTAKBANK, RELIANCE, TCS")
    matches = universe[
        universe["symbol"].str.contains(query, case=False, na=False)
        | universe["company_name"].str.contains(query, case=False, na=False)
    ] if query else universe.head(30)

    selected = st.selectbox(
        "Select stock",
        matches["symbol"].tolist() if not matches.empty else ["No match"],
    )

    if selected != "No match":
        company = universe.loc[universe["symbol"] == selected, "company_name"].iloc[0]
        st.caption(company)

        if st.button("Analyze selected stock", type="primary"):
            with st.spinner(f"Analyzing {selected}..."):
                row = scan_single(selected, max_distance=max_distance, period=period)
            if row is None:
                st.error("No usable market data was returned for this symbol.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Setup rating", f"{row['rating']}/10")
                c2.metric("Price", f"₹{row['price']:,.2f}")
                c3.metric("Breakout level", f"₹{row['breakout_level']:,.2f}")
                c4.metric("Distance", f"{row['distance_pct']:.2f}%")

                st.write(
                    f"**Status:** {row['status']}  •  "
                    f"**RSI:** {row['rsi']:.1f}  •  "
                    f"**ADX:** {row['adx']:.1f}  •  "
                    f"**RVOL:** {row['rvol']:.2f}x  •  "
                    f"**Candle:** {row['candlestick']}"
                )

                hist = fetch_history(selected, period=period)
                if hist is not None and not hist.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=hist.index,
                        open=hist["Open"],
                        high=hist["High"],
                        low=hist["Low"],
                        close=hist["Close"],
                        name=selected,
                    ))
                    fig.add_hline(
                        y=row["breakout_level"],
                        line_dash="dash",
                        annotation_text=f"Breakout ₹{row['breakout_level']:.2f}",
                    )
                    fig.update_layout(
                        height=600,
                        xaxis_rangeslider_visible=False,
                        title=f"{selected} — Daily Candlestick Chart",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Why this setup scored this way")
                st.write(row["explanation"])


with tab3:
    st.subheader("🧪 Historical Backtest")
    st.caption(
        "Walk-forward test: the signal uses only data available on that historical date. "
        "Future bars are used only to measure what happened afterward."
    )

    bt_symbol = st.selectbox(
        "Stock to backtest",
        symbols if 'symbols' in locals() else universe["symbol"].tolist()
    )
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        bt_period = st.selectbox("Backtest history", ["3y", "5y"], index=1)
    with bc2:
        bt_rating = st.slider("Minimum rating", 5, 9, 7)
    with bc3:
        bt_hold = st.slider("Forward holding window (days)", 3, 20, 10)

    if st.button("Run Backtest", type="primary"):
        from scanner import backtest_symbol
        with st.spinner("Running walk-forward backtest..."):
            bt = backtest_symbol(
                bt_symbol, period=bt_period,
                min_rating=bt_rating, hold_days=bt_hold
            )
        summary = bt["summary"]
        trades = bt["trades"]

        if not summary or summary.get("signals", 0) == 0:
            st.warning("No historical signals matched these settings.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Signals", summary["signals"])
            m2.metric("Breakout hit rate", f"{summary['breakout_hit_rate_pct']:.1f}%")
            m3.metric("Avg final return", f"{summary['avg_final_return_pct']:.2f}%")
            m4.metric("Avg max return", f"{summary['avg_max_return_pct']:.2f}%")

            st.write(
                f"Median final return: **{summary['median_final_return_pct']:.2f}%** • "
                f"5% downside-touch rate: **{summary['stop_5pct_rate_pct']:.1f}%**"
            )

            st.dataframe(
                trades.sort_values("date", ascending=False),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download backtest trades",
                trades.to_csv(index=False).encode("utf-8"),
                f"{bt_symbol}_backtest.csv",
                "text/csv"
            )



with st.expander("⚙️ Tune scoring weights from historical data"):
    st.write(
        "This tests a small set of predefined, non-extreme weight combinations across "
        "historical NIFTY 200 samples. It is intended to reduce arbitrary weighting, "
        "not to guarantee future performance."
    )
    opt_symbols = st.multiselect(
        "Stocks used for tuning (use a diversified sample)",
        universe["symbol"].tolist(),
        default=universe["symbol"].tolist()[:10],
    )
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        opt_period = st.selectbox("Tuning history", ["3y", "5y"], index=1)
    with oc2:
        opt_rating = st.slider("Minimum rating for test", 6, 9, 7, key="opt_rating")
    with oc3:
        opt_hold = st.slider("Forward days", 5, 20, 10, key="opt_hold")

    if st.button("🧠 Tune scoring formula"):
        if len(opt_symbols) < 3:
            st.warning("Select at least 3 stocks for a useful tuning sample.")
        else:
            from scanner import optimize_weights
            with st.spinner("Testing scoring configurations..."):
                ranking, best_weights = optimize_weights(
                    opt_symbols, period=opt_period,
                    min_rating=opt_rating, hold_days=opt_hold,
                    max_symbols=len(opt_symbols)
                )
            if ranking.empty:
                st.error("Not enough historical signals to tune the weights.")
            else:
                st.success("Best candidate selected from the tested configurations.")
                st.json(best_weights)
                st.dataframe(
                    ranking.drop(columns=["weights"]),
                    use_container_width=True,
                    hide_index=True
                )
                st.info(
                    "The optimizer does not automatically overwrite the production "
                    "weights. This prevents accidental overfitting. Copy the selected "
                    "weights into DEFAULT_WEIGHTS only after reviewing the sample."
                )


st.markdown("---")
st.caption(
    "Important: A 10/10 score means the technical setup is closest to the scanner's "
    "defined breakout conditions. It does not mean a breakout will happen or that a "
    "stock will rise. Validate signals and data independently."
)
