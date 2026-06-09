import numpy as np
import pandas as pd


def make_sample_history(length: int = 80) -> pd.DataFrame:
    close = np.linspace(100.0, 120.0, length)
    data = {
        "Close": close,
        "Open": close - 0.5,
        "High": close + 0.5,
        "Low": close - 1.0,
        "Volume": np.linspace(1000, 2000, length),
    }
    df = pd.DataFrame(data)
    df["MA5"] = df["Close"].rolling(window=5, min_periods=1).mean()
    df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["MA5_vs_MA20"] = (df["MA5"] - df["MA20"]) / df["MA20"].replace(0, np.nan)
    df["MA20_vs_MA50"] = (df["MA20"] - df["MA50"]) / df["MA50"].replace(0, np.nan)
    df["RSI14"] = 50 + np.sin(np.linspace(0, 3.0, length)) * 20
    df["BB_Position"] = np.linspace(0.2, 0.8, length)
    df["MACD_Cross"] = np.where(np.arange(length) % 5 == 0, 1, 0)
    df["Momentum_Normalized"] = np.gradient(df["Close"]) / df["Close"].shift(1).replace(0, np.nan)
    df["ATR_Pct"] = np.abs(df["Close"].diff()).fillna(0) / df["Close"] * 100
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df


def make_volatile_history(length: int = 80) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    close = 100 + rng.standard_normal(length).cumsum()
    data = {
        "Close": close,
        "Open": close + rng.standard_normal(length) * 0.5,
        "High": close + np.abs(rng.standard_normal(length) * 1.0),
        "Low": close - np.abs(rng.standard_normal(length) * 1.0),
        "Volume": 1000 + np.abs(rng.standard_normal(length) * 300),
    }
    df = pd.DataFrame(data)
    df["MA5"] = df["Close"].rolling(window=5, min_periods=1).mean()
    df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["MA5_vs_MA20"] = (df["MA5"] - df["MA20"]) / df["MA20"].replace(0, np.nan)
    df["MA20_vs_MA50"] = (df["MA20"] - df["MA50"]) / df["MA50"].replace(0, np.nan)
    df["RSI14"] = 50 + np.sin(np.linspace(0, 12.0, length)) * 25
    df["BB_Position"] = np.where(np.arange(length) % 3 == 0, 0.02, 0.98)
    df["MACD_Cross"] = np.where(np.arange(length) % 2 == 0, 1, -1)
    df["Momentum_Normalized"] = np.gradient(df["Close"]) / df["Close"].shift(1).replace(0, np.nan)
    df["ATR_Pct"] = np.abs(df["Close"].diff()).fillna(0) / df["Close"] * 100
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df


def test_get_feature_columns_includes_sentiment_only_when_present():
    from app.models.predictor import get_feature_columns

    df = pd.DataFrame({"RSI14": [50.0], "Sentiment_Score": [0.2]})
    columns = get_feature_columns(df, include_sentiment=True)
    assert "RSI14" in columns
    assert "Sentiment_Score" in columns

    columns = get_feature_columns(df, include_sentiment=False)
    assert "RSI14" in columns
    assert "Sentiment_Score" not in columns


def test_prepare_training_data_returns_arrays_for_valid_history():
    from app.models.predictor import prepare_training_data

    df = make_sample_history(80)
    feature_cols = ["RSI14", "MA5_vs_MA20", "MA20_vs_MA50", "ATR_Pct"]
    X, y, index = prepare_training_data(df, feature_cols, min_samples=10)

    assert X is not None and y is not None and index is not None
    assert X.shape[0] == len(y)
    assert X.shape[1] == len(feature_cols)
    assert X.dtype == np.float32
    assert y.dtype in (np.int32, np.int64)
    assert np.isfinite(X).all()
    assert not np.isnan(y).any()


def test_prepare_training_data_handles_nan_and_inf_values():
    from app.models.predictor import prepare_training_data

    df = make_sample_history(80)
    df.loc[5, "RSI14"] = np.nan
    df.loc[6, "ATR_Pct"] = np.inf
    feature_cols = ["RSI14", "MA5_vs_MA20", "MA20_vs_MA50", "ATR_Pct"]

    X, y, index = prepare_training_data(df, feature_cols, min_samples=10)

    assert X is not None and y is not None
    assert np.isfinite(X).all()
    assert (X >= -10).all() and (X <= 10).all()


def test_prepare_training_data_returns_none_for_short_history():
    from app.models.predictor import prepare_training_data

    df = make_sample_history(10)
    feature_cols = ["RSI14", "MA5_vs_MA20", "MA20_vs_MA50", "ATR_Pct"]
    X, y, index = prepare_training_data(df, feature_cols, min_samples=20)

    assert X is None
    assert y is None
    assert index is None


def test_regime_detector_returns_regime_and_use_lstm_flags():
    from app.models.predictor import RegimeDetector

    simple_df = make_sample_history(80)
    detector = RegimeDetector(lookback=40)
    simple_scores = detector.compute(simple_df)

    assert simple_scores["regime"] in {"simple", "moderate", "complex"}
    assert isinstance(simple_scores["use_lstm"], bool)
    assert 0.0 <= simple_scores["complexity"] <= 1.0
    expected_keys = {
        "volatility", "trend_inconsistency", "rsi_extremes",
        "macd_cross_freq", "momentum_reversal", "bb_breakout",
    }
    assert set(simple_scores["scores"]) >= expected_keys

    volatile_df = make_volatile_history(80)
    volatile_scores = detector.compute(volatile_df)
    assert volatile_scores["complexity"] >= simple_scores["complexity"]
    assert (
        volatile_scores["use_lstm"] is True
        or volatile_scores["regime"] in {"moderate", "complex"}
    )


def test_build_result_produces_expected_signal_mapping():
    from app.models.predictor import _build_result

    buy = _build_result(0.70, "XGBoost")
    assert buy["signal"] == "BUY"
    assert buy["direction"] == 1
    assert buy["up_probability"] == 0.70

    sell = _build_result(0.38, "XGBoost")
    assert sell["signal"] == "SELL"
    assert sell["direction"] == 0
    assert sell["down_probability"] == 0.62

    hold = _build_result(0.53, "XGBoost")
    assert hold["signal"] == "HOLD"
    assert hold["confidence"] == 0.53


def test_run_backtest_returns_portfolio_for_dummy_predictor():
    from app.models.predictor import run_backtest

    df = make_sample_history(85)
    df.index = pd.date_range("2025-01-01", periods=len(df), freq="D")

    class DummyPredictor:
        def __init__(self):
            self.is_trained = True

        def train(self, df, include_sentiment=False):
            self.is_trained = True
            return {}

        def predict(self, df):
            return {"signal": "BUY"}

    predictor = DummyPredictor()
    backtest = run_backtest(df, predictor, initial_capital=10000, commission_rate=0.0)

    assert backtest["initial_capital"] == 10000
    assert "final_capital" in backtest
    assert "portfolio_values" in backtest
    assert backtest["n_trades"] >= 0
    expected_return = round((backtest["final_capital"] / 10000 - 1) * 100, 2)
    assert backtest["strategy_return_pct"] == expected_return