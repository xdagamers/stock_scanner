import sys, types
fake = types.ModuleType("yfinance")
fake.download = lambda *a, **k: None
sys.modules["yfinance"] = fake

import numpy as np
import pandas as pd
import scanner


def make_data(n=360, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = pd.Series(100*np.exp(np.cumsum(rng.normal(.0002,.01,n))), index=idx)
    op = close.shift().fillna(close.iloc[0])
    hi = pd.concat([op, close], axis=1).max(axis=1) * (1+rng.uniform(.001,.01,n))
    lo = pd.concat([op, close], axis=1).min(axis=1) * (1-rng.uniform(.001,.01,n))
    vol = pd.Series(rng.integers(100_000,200_000,n), index=idx)
    return pd.DataFrame({"Open":op,"High":hi,"Low":lo,"Close":close,"Volume":vol})


def test_indicators():
    df=make_data()
    assert len(scanner.rsi(df.Close)) == len(df)
    assert len(scanner.atr(df)) == len(df)
    assert all(len(x)==len(df) for x in scanner.adx(df))


def test_score_range_and_weights():
    df=make_data()
    row=scanner.score_stock(df)
    assert row is not None
    assert 0 <= row["score"] <= 100
    assert 1 <= row["rating"] <= 10
    assert sum(scanner.DEFAULT_WEIGHTS.values()) == 100


def test_backtest_walk_forward():
    df=make_data(500)
    t=scanner._backtest_with_weights(df, scanner.DEFAULT_WEIGHTS, min_rating=5, hold_days=10)
    assert isinstance(t, pd.DataFrame)
    if not t.empty:
        assert t["date"].is_monotonic_increasing


def test_optimizer():
    df=make_data(360)
    old=scanner.fetch_history
    scanner.fetch_history=lambda symbol, period="2y": df
    ranking,best=scanner.optimize_weights(["A","B","C"], max_symbols=3, period="1y", min_rating=5, hold_days=5)
    scanner.fetch_history=old
    assert len(ranking) > 0
    assert sum(best.values()) == 100


if __name__ == "__main__":
    test_indicators(); test_score_range_and_weights(); test_backtest_walk_forward(); test_optimizer()
    print("ALL TESTS PASSED")
