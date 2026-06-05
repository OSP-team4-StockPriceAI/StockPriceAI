import numpy as np
import pandas as pd

from app.pipelines.technical import add_all_indicators, get_current_signals, get_support_resistance
from app.pipelines.fetcher import is_korean_ticker, normalize_ticker
from app.models.predictor import _bar, _build_result
from app.pipelines.scanner import dp_blend


def make_sample_df(n: int = 60) -> pd.DataFrame:
    dates = pd.date_range(end=pd.Timestamp("2025-01-01"), periods=n, freq="D")
    close = np.linspace(100.0, 120.0, n) + np.random.RandomState(0).randn(n) * 0.1
    open_ = close - np.random.RandomState(1).randn(n) * 0.1
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1
    vol = np.abs(np.random.RandomState(2).randn(n)) * 1000 + 100
    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
    }, index=dates)
    return df


def test_technical_indicators_and_utils():
    df = make_sample_df(80)
    df2 = add_all_indicators(df)
    # basic expectations
    assert "RSI14" in df2.columns
    assert "MACD" in df2.columns
    assert df2["Close"].dtype == "float32"

    signals = get_current_signals(df2)
    assert isinstance(signals, dict)

    sr = get_support_resistance(df2)
    assert "current" in sr and "resistance_20d" in sr

    # small utility checks
    assert is_korean_ticker("005930") is True
    assert normalize_ticker("005930") == "005930.KS"

    bar = _bar(5, 10)
    assert isinstance(bar, str) and "/" in bar

    res = _build_result(0.6, "tst")
    assert res["model"] == "tst"


def test_dp_blend_behaviour():
    old = {"up_probability": 0.4, "composite_score": 0.5, "dp_blend_count": 1}
    new = {"up_probability": 0.6, "composite_score": 0.8}
    blended = dp_blend(old, new, alpha=0.5)
    assert blended["dp_blend_count"] == 2
    assert 0.4 <= blended["up_probability"] <= 0.6
