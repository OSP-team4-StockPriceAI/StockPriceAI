"""기술적 지표 서비스 단위 테스트"""

import numpy as np
import pandas as pd

from app.pipelines.technical import (
    add_all_indicators,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
    get_current_signals,
)


def _make_df(n: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.abs(np.random.randn(n) * 1e6 + 1e7),
        },
        index=idx,
    )


def test_sma():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    sma = calculate_sma(s, 3)
    assert abs(float(sma.iloc[-1]) - 4.0) < 1e-6


def test_ema_shape():
    s = pd.Series(range(50), dtype=float)
    ema = calculate_ema(s, 10)
    assert len(ema) == 50


def test_rsi_range():
    s = pd.Series(range(100), dtype=float)
    rsi = calculate_rsi(s, 14)
    assert rsi.between(0, 100).all()


def test_macd_returns_three():
    s = pd.Series(range(100), dtype=float)
    macd, signal, hist = calculate_macd(s)
    assert len(macd) == len(signal) == len(hist) == 100


def test_add_all_indicators_columns():
    df = _make_df(300)
    result = add_all_indicators(df)
    required = [
        "RSI14", "MACD", "BB_Upper", "BB_Lower", "ATR14", "STOCH_K", "OBV", "Target",
        "ADX14", "Plus_DI", "Minus_DI", "MA_Alignment_Spread",
        "MA_Bias_Short", "MA_Bias_Mid", "Regime_Alignment",
        "Regime_Directional", "Regime_Trend_Strength",
        "ADX_Momentum", "DI_Diff",
        "Regime_Prob_Bull", "Regime_Prob_Bear", "Regime_Prob_Sideways",
    ]
    for col in required:
        assert col in result.columns, f"Missing column: {col}"


def test_get_current_signals_keys():
    df = _make_df(300)
    df = add_all_indicators(df)
    signals = get_current_signals(df)
    assert "RSI" in signals
    assert "MACD" in signals
    assert "Bollinger" in signals
    assert "MA" in signals
    for key, val in signals.items():
        assert len(val) == 3
        assert val[0] in ("BUY", "SELL", "HOLD", "WATCH")