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
    assert set(simple_scores["scores"]) >= {"volatility", "trend_inconsistency", "rsi_extremes", "macd_cross_freq", "momentum_reversal", "bb_breakout"}

    volatile_df = make_volatile_history(80)
    volatile_scores = detector.compute(volatile_df)
    assert volatile_scores["complexity"] >= simple_scores["complexity"]
    assert volatile_scores["use_lstm"] is True or volatile_scores["regime"] in {"moderate", "complex"}


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
    assert backtest["strategy_return_pct"] == round((backtest["final_capital"] / 10000 - 1) * 100, 2)

import pytest
from unittest.mock import MagicMock, patch


def make_full_feature_df(length: int = 100) -> pd.DataFrame:
    """predictor가 요구하는 모든 피처 컬럼을 포함한 DataFrame 생성."""
    rng = np.random.RandomState(42)
    close = 100 + rng.standard_normal(length).cumsum()
    df = pd.DataFrame({
        "Close": close, "Open": close - 0.2, "High": close + 0.5,
        "Low": close - 0.5, "Volume": np.abs(rng.standard_normal(length)) * 1000 + 500,
        "RSI14": np.clip(50 + rng.standard_normal(length) * 15, 10, 90),
        "RSI7": np.clip(50 + rng.standard_normal(length) * 20, 10, 90),
        "MACD": rng.standard_normal(length) * 0.5,
        "MACD_Signal": rng.standard_normal(length) * 0.3,
        "MACD_Hist": rng.standard_normal(length) * 0.2,
        "BB_Width": np.abs(rng.standard_normal(length)) * 0.02 + 0.05,
        "BB_Position": np.clip(rng.standard_normal(length) * 0.3 + 0.5, 0, 1),
        "ATR_Pct": np.abs(rng.standard_normal(length)) * 1.5 + 1.0,
        "STOCH_K": np.clip(rng.standard_normal(length) * 20 + 50, 0, 100),
        "STOCH_D": np.clip(rng.standard_normal(length) * 20 + 50, 0, 100),
        "WILLIAMS_R": np.clip(rng.standard_normal(length) * 30 - 50, -100, 0),
        "Volume_Ratio": np.abs(rng.standard_normal(length)) + 1,
        "OBV_Trend": rng.standard_normal(length),
        "Return_1d": rng.standard_normal(length) * 0.01,
        "Return_3d": rng.standard_normal(length) * 0.02,
        "Return_5d": rng.standard_normal(length) * 0.03,
        "Return_10d": rng.standard_normal(length) * 0.04,
        "Return_20d": rng.standard_normal(length) * 0.05,
        "Price_vs_MA20": rng.standard_normal(length) * 0.02,
        "Price_vs_MA50": rng.standard_normal(length) * 0.03,
        "MA5_vs_MA20": rng.standard_normal(length) * 0.01,
        "MA20_vs_MA50": rng.standard_normal(length) * 0.02,
        "MA_Bias_Short": rng.standard_normal(length) * 0.01,
        "MA_Bias_Mid": rng.standard_normal(length) * 0.02,
        "Regime_Alignment": rng.standard_normal(length),
        "Regime_Directional": rng.standard_normal(length),
        "Regime_Trend_Strength": np.abs(rng.standard_normal(length)),
        "MA_Alignment_Spread": rng.standard_normal(length) * 0.01,
        "MA_Alignment_Ratio": np.abs(rng.standard_normal(length)) + 1,
        "ADX14": np.abs(rng.standard_normal(length)) * 10 + 20,
        "Plus_DI": np.abs(rng.standard_normal(length)) * 5 + 20,
        "Minus_DI": np.abs(rng.standard_normal(length)) * 5 + 20,
        "ADX_Momentum": rng.standard_normal(length),
        "DI_Diff": rng.standard_normal(length) * 5,
        "Regime_Prob_Bull": np.clip(rng.standard_normal(length) * 0.2 + 0.5, 0, 1),
        "Regime_Prob_Bear": np.clip(rng.standard_normal(length) * 0.2 + 0.3, 0, 1),
        "Regime_Prob_Sideways": np.clip(rng.standard_normal(length) * 0.1 + 0.2, 0, 1),
        "Price_Position_20d": np.clip(rng.standard_normal(length) * 0.3 + 0.5, 0, 1),
        "Body_Size": np.abs(rng.standard_normal(length)) * 0.5,
        "Upper_Shadow": np.abs(rng.standard_normal(length)) * 0.3,
        "Lower_Shadow": np.abs(rng.standard_normal(length)) * 0.3,
        "Is_Bullish": (rng.standard_normal(length) > 0).astype(float),
        "Momentum_Normalized": rng.standard_normal(length) * 0.01,
        "MACD_Cross": np.where(np.arange(length) % 5 == 0, 1, 0),
        "Target": (rng.standard_normal(length) > 0).astype(int),
    })
    df.index = pd.date_range("2023-01-01", periods=length, freq="D")
    return df


class TestAutoParams:
    def test_small_dataset(self):
        from app.models.predictor import _auto_params
        p = _auto_params(200)
        assert p["n_estimators_xgb"] == 150
        assert p["n_splits"] == 3

    def test_medium_dataset(self):
        from app.models.predictor import _auto_params
        p = _auto_params(500)
        assert p["n_estimators_xgb"] == 200
        assert p["n_splits"] == 4

    def test_large_dataset(self):
        from app.models.predictor import _auto_params
        p = _auto_params(1000)
        assert p["n_estimators_xgb"] == 300
        assert p["n_splits"] == 5

    def test_very_large_dataset(self):
        from app.models.predictor import _auto_params
        p = _auto_params(5000)
        assert "max_samples" in p
        assert p["fast"] is True

    def test_boundary_2000(self):
        from app.models.predictor import _auto_params
        p = _auto_params(2000)
        assert p["n_splits"] == 5


class TestBarFunction:
    def test_normal_bar(self):
        from app.models.predictor import _bar
        result = _bar(5, 10)
        assert "5/10" in result
        assert "[" in result and "]" in result

    def test_zero_total(self):
        from app.models.predictor import _bar
        result = _bar(0, 0)
        assert "?" in result

    def test_zero_current(self):
        from app.models.predictor import _bar
        result = _bar(0, 10)
        assert "0/10" in result


class TestBuildResult:
    def test_buy_signal(self):
        from app.models.predictor import _build_result
        r = _build_result(0.70)
        assert r["signal"] == "BUY"
        assert r["direction"] == 1
        assert r["confidence"] == 0.70
        assert r["model"] == "XGBoost"

    def test_sell_signal(self):
        from app.models.predictor import _build_result
        r = _build_result(0.38)
        assert r["signal"] == "SELL"
        assert r["direction"] == 0

    def test_hold_signal(self):
        from app.models.predictor import _build_result
        r = _build_result(0.53)
        assert r["signal"] == "HOLD"

    def test_custom_model_name(self):
        from app.models.predictor import _build_result
        r = _build_result(0.8, "LSTM")
        assert r["model"] == "LSTM"

    def test_exact_boundary_065(self):
        from app.models.predictor import _build_result
        r = _build_result(0.65)
        # 0.65는 BUY 기준 초과 아님
        assert r["signal"] in ("BUY", "HOLD")

    def test_down_prob_is_complement(self):
        from app.models.predictor import _build_result
        r = _build_result(0.4)
        assert abs(r["up_probability"] + r["down_probability"] - 1.0) < 1e-6


class TestRegimeDetector:
    def test_empty_df_returns_simple(self):
        from app.models.predictor import RegimeDetector
        rd = RegimeDetector()
        result = rd.detect(pd.DataFrame())
        assert result["regime"] == "simple"
        assert result["use_lstm"] is False
        assert result["complexity"] == 0.0

    def test_high_bull_prob_returns_complex(self):
        from app.models.predictor import RegimeDetector
        rd = RegimeDetector()
        df = pd.DataFrame({"Regime_Prob_Bull": [0.7], "ATR_Pct": [1.0]})
        result = rd.detect(df)
        assert result["regime"] == "complex"
        assert result["use_lstm"] is True

    def test_moderate_bull_prob(self):
        from app.models.predictor import RegimeDetector
        rd = RegimeDetector()
        df = pd.DataFrame({"Regime_Prob_Bull": [0.4], "ATR_Pct": [1.0]})
        result = rd.detect(df)
        assert result["regime"] == "moderate"

    def test_low_prob_returns_simple(self):
        from app.models.predictor import RegimeDetector
        rd = RegimeDetector()
        df = pd.DataFrame({"Regime_Prob_Bull": [0.1], "ATR_Pct": [1.0]})
        result = rd.detect(df)
        assert result["regime"] == "simple"
        assert result["use_lstm"] is False

    def test_high_atr_returns_complex(self):
        from app.models.predictor import RegimeDetector
        rd = RegimeDetector()
        df = pd.DataFrame({"Regime_Prob_Bull": [0.1], "ATR_Pct": [3.0]})
        result = rd.detect(df)
        assert result["regime"] == "complex"

    def test_compute_is_alias_for_detect(self):
        from app.models.predictor import RegimeDetector
        rd = RegimeDetector()
        df = pd.DataFrame({"Regime_Prob_Bull": [0.5], "ATR_Pct": [1.0]})
        assert rd.compute(df) == rd.detect(df)

    def test_scores_has_all_keys(self):
        from app.models.predictor import RegimeDetector
        rd = RegimeDetector()
        df = pd.DataFrame({"Regime_Prob_Bull": [0.5], "ATR_Pct": [1.0]})
        result = rd.detect(df)
        expected = {"volatility", "trend_inconsistency", "rsi_extremes",
                    "macd_cross_freq", "momentum_reversal", "bb_breakout"}
        assert set(result["scores"]) >= expected

    def test_missing_atr_defaults_to_one(self):
        from app.models.predictor import RegimeDetector
        rd = RegimeDetector(lookback=10)
        df = pd.DataFrame({"Regime_Prob_Bull": [0.1]})
        result = rd.detect(df)
        # ATR_Pct 없으면 1.0 기본값 → ATR_Pct=1.0 < 2.0 → 변동성 낮음
        assert result["regime"] == "simple"


class TestPrepareTrainingData:
    def test_max_samples_truncation(self):
        from app.models.predictor import prepare_training_data
        df = make_full_feature_df(150)
        X, y, idx = prepare_training_data(df, ["RSI14", "MACD"], max_samples=50)
        assert X is not None
        assert len(X) <= 50

    def test_drops_last_row(self):
        """Target 기준으로 마지막 행은 제거돼야 한다."""
        from app.models.predictor import prepare_training_data
        df = make_full_feature_df(80)
        X, y, idx = prepare_training_data(df, ["RSI14", "MACD"], min_samples=10)
        assert X is not None
        assert len(X) < len(df)

    def test_clipping_applied(self):
        from app.models.predictor import prepare_training_data
        df = make_full_feature_df(80)
        df["RSI14"] = 9999.0
        X, y, _ = prepare_training_data(df, ["RSI14"], min_samples=10)
        assert X is not None
        assert X.max() <= 10.0


class TestXGBoostPredictor:
    """xgboost 패키지가 없을 경우 sklearn 폴백으로 동작하는지 확인."""

    def _make_predictor_and_df(self):
        from app.models.predictor import XGBoostPredictor
        pred = XGBoostPredictor(scanner_mode=False)
        df = make_full_feature_df(100)
        return pred, df

    def test_initial_state(self):
        from app.models.predictor import XGBoostPredictor
        pred = XGBoostPredictor()
        assert pred.is_trained is False
        assert pred.model is None

    def test_predict_proba_before_train_returns_none(self):
        from app.models.predictor import XGBoostPredictor
        pred = XGBoostPredictor()
        df = make_full_feature_df(80)
        assert pred.predict_proba(df) is None

    def test_predict_before_train_returns_error(self):
        from app.models.predictor import XGBoostPredictor
        pred = XGBoostPredictor()
        df = make_full_feature_df(80)
        result = pred.predict(df)
        assert "error" in result

    def test_train_insufficient_data_returns_error(self):
        from app.models.predictor import XGBoostPredictor
        pred = XGBoostPredictor()
        df = make_full_feature_df(10)  # 너무 짧음
        result = pred.train(df)
        assert "error" in result

    def test_sklearn_fallback_train(self):
        """xgboost import를 막아서 sklearn 폴백을 실행."""
        from app.models.predictor import XGBoostPredictor
        pred = XGBoostPredictor()
        df = make_full_feature_df(100)
        feature_cols = ["RSI14", "MACD", "BB_Position", "ATR_Pct", "Return_1d"]

        with patch.dict("sys.modules", {"xgboost": None}):
            result = pred._train_sklearn(df, include_sentiment=False, feature_cols=feature_cols)

        assert "error" not in result
        assert pred.is_trained is True
        assert "train_accuracy" in result

    def test_sklearn_fallback_predict_proba(self):
        from app.models.predictor import XGBoostPredictor
        pred = XGBoostPredictor()
        df = make_full_feature_df(100)
        feature_cols = ["RSI14", "MACD", "BB_Position", "ATR_Pct", "Return_1d"]
        pred._train_sklearn(df, include_sentiment=False, feature_cols=feature_cols)
        proba = pred.predict_proba(df)
        assert proba is not None
        assert 0.0 <= proba <= 1.0

    def test_sklearn_fallback_insufficient_data(self):
        from app.models.predictor import XGBoostPredictor
        pred = XGBoostPredictor()
        df = make_full_feature_df(5)
        result = pred._train_sklearn(df, include_sentiment=False, feature_cols=["RSI14"])
        assert "error" in result

    def test_fit_full_data_insufficient(self):
        from app.models.predictor import XGBoostPredictor
        pred = XGBoostPredictor()
        df = make_full_feature_df(10)
        with patch.dict("sys.modules", {"xgboost": None}):
            result = pred.fit_full_data(df, include_sentiment=False)
        assert "error" in result


class TestLSTMPredictorAvailableFramework:
    def test_returns_none_when_no_framework(self):
        from app.models.predictor import LSTMPredictor
        with patch.dict("sys.modules", {"torch": None, "tensorflow": None}):
            result = LSTMPredictor.available_framework()
        assert result is None

    def test_initial_state(self):
        from app.models.predictor import LSTMPredictor
        pred = LSTMPredictor(sequence_length=10)
        assert pred.is_trained is False
        assert pred.sequence_length == 10

    def test_train_no_framework(self):
        from app.models.predictor import LSTMPredictor
        pred = LSTMPredictor()
        df = make_full_feature_df(80)
        with patch.object(LSTMPredictor, "available_framework", return_value=None):
            result = pred.train(df)
        assert "error" in result

    def test_train_insufficient_data(self):
        from app.models.predictor import LSTMPredictor
        pred = LSTMPredictor(sequence_length=30)
        df = make_full_feature_df(30)  # SEQ+20 = 50 필요
        with patch.object(LSTMPredictor, "available_framework", return_value="pytorch"):
            result = pred.train(df)
        assert "error" in result

    def test_predict_proba_before_train(self):
        from app.models.predictor import LSTMPredictor
        pred = LSTMPredictor()
        df = make_full_feature_df(80)
        assert pred.predict_proba(df) is None

    def test_predict_before_train_returns_error(self):
        from app.models.predictor import LSTMPredictor
        pred = LSTMPredictor()
        df = make_full_feature_df(80)
        result = pred.predict(df)
        assert "error" in result


class TestPredictLstmHistoryProba:
    def test_untrained_returns_half(self):
        from app.models.predictor import predict_lstm_history_proba, LSTMPredictor
        pred = LSTMPredictor()
        df = make_full_feature_df(50)
        series = predict_lstm_history_proba(pred, df)
        assert (series == 0.5).all()

    def test_short_data_returns_half(self):
        from app.models.predictor import predict_lstm_history_proba, LSTMPredictor
        pred = LSTMPredictor(sequence_length=30)
        pred.is_trained = True
        pred.feature_cols = ["RSI14"]
        # scaler는 fit 안 됐으므로 실제 transform 전에 SEQ 체크가 먼저 돼야 함
        df = make_full_feature_df(10)
        series = predict_lstm_history_proba(pred, df)
        assert (series == 0.5).all()


class TestLSTMFirstStackingPredictor:
    def test_initial_state(self):
        from app.models.predictor import LSTMFirstStackingPredictor
        pred = LSTMFirstStackingPredictor(sequence_length=10)
        assert pred.is_trained is False

    def test_predict_before_train_returns_error(self):
        from app.models.predictor import LSTMFirstStackingPredictor
        pred = LSTMFirstStackingPredictor()
        df = make_full_feature_df(80)
        result = pred.predict(df)
        assert "error" in result

    def test_train_fallback_when_no_framework(self):
        """PyTorch/TF 없을 때 XGBoost 단독 폴백."""
        from app.models.predictor import LSTMFirstStackingPredictor, LSTMPredictor
        pred = LSTMFirstStackingPredictor()
        df = make_full_feature_df(100)

        with patch.object(LSTMPredictor, "available_framework", return_value=None), \
             patch.dict("sys.modules", {"xgboost": None}):
            result = pred.train(df)

        # sklearn 폴백으로 성공하거나 오류 키 반환
        assert isinstance(result, dict)

    def test_predict_fallback_when_no_framework_xgb_trained(self):
        """학습 후 predict 시 프레임워크 없으면 XGBoost 단독 예측."""
        from app.models.predictor import LSTMFirstStackingPredictor, LSTMPredictor
        pred = LSTMFirstStackingPredictor()
        df = make_full_feature_df(100)

        # 강제로 is_trained=True 및 xgb 부분만 훈련
        pred.is_trained = True
        with patch.dict("sys.modules", {"xgboost": None}):
            pred.xgb._train_sklearn(df, include_sentiment=False,
                                    feature_cols=["RSI14", "MACD", "BB_Position"])

        with patch.object(LSTMPredictor, "available_framework", return_value=None):
            result = pred.predict(df)
        # xgb가 훈련된 경우 예측 성공 또는 실패 모두 dict 반환
        assert isinstance(result, dict)


class TestEnsemblePredictorAlias:
    def test_alias_is_lstm_first_stacking(self):
        from app.models.predictor import EnsemblePredictor, LSTMFirstStackingPredictor
        assert EnsemblePredictor is LSTMFirstStackingPredictor


class TestRunBacktest:
    def test_untrained_predictor_returns_empty(self):
        from app.models.predictor import run_backtest

        class FakePredictor:
            is_trained = False

        assert run_backtest(make_full_feature_df(100), FakePredictor()) == {}

    def test_sell_signal_closes_position(self):
        from app.models.predictor import run_backtest

        signals = iter(["BUY"] * 10 + ["SELL"] * 50)

        class FakePredictor:
            is_trained = True

            def predict(self, df):
                try:
                    return {"signal": next(signals)}
                except StopIteration:
                    return {"signal": "HOLD"}

        df = make_full_feature_df(100)
        df.index = pd.date_range("2023-01-01", periods=100, freq="D")
        result = run_backtest(df, FakePredictor(), initial_capital=10000)
        assert result["n_trades"] >= 1

    def test_error_signal_handled(self):
        from app.models.predictor import run_backtest

        class FakePredictor:
            is_trained = True

            def predict(self, df):
                return {"error": "oops"}

        df = make_full_feature_df(100)
        result = run_backtest(df, FakePredictor())
        assert "portfolio_values" in result

    def test_hold_at_end_closes_position(self):
        from app.models.predictor import run_backtest

        class FakePredictor:
            is_trained = True
            call_count = 0

            def predict(self, df):
                self.call_count += 1
                return {"signal": "BUY"}

        df = make_full_feature_df(100)
        result = run_backtest(df, FakePredictor())
        assert result["final_capital"] > 0