"""GET /api/v1/technical/{ticker} 엔드포인트 테스트"""

from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


def _make_df(n: int = 100) -> pd.DataFrame:
    close = np.linspace(100.0, 150.0, n)
    df = pd.DataFrame({
        "Close": close, "Open": close - 1,
        "High": close + 1, "Low": close - 1, "Volume": np.full(n, 1e6),
    })
    for col in ["RSI14", "RSI7", "MACD", "MACD_Signal", "MACD_Hist",
                "BB_Upper", "BB_Middle", "BB_Lower", "BB_Position",
                "STOCH_K", "STOCH_D", "WILLIAMS_R", "ATR14", "ATR_Pct",
                "Volume_Ratio", "OBV_Trend", "MA5", "MA20", "MA50", "MA200",
                "Momentum_Normalized", "Target"]:
        df[col] = 0.5
    return df


_SIGNALS = {
    "RSI": ("BUY", "RSI oversold", "green"),
    "MACD": ("SELL", "MACD bearish", "red"),
    "Bollinger": ("HOLD", "BB neutral", "gray"),
    "MA": ("BUY", "MA bullish", "green"),
}
_SR = {"support": 140.0, "resistance": 160.0}
_INFO = {"sector": "Technology", "longName": "Apple Inc.", "currency": "USD"}


# ── 정상 조회 ─────────────────────────────────────────────────

def test_get_technical_success_returns_200(client: TestClient) -> None:
    """정상 요청 시 TechnicalResponse 스키마에 맞는 200 응답을 반환한다."""
    df = _make_df()

    with patch("app.pipelines.fetcher.fetch_stock_data", return_value=(df, _INFO)), \
         patch("app.pipelines.technical.add_all_indicators", return_value=df), \
         patch("app.pipelines.technical.get_current_signals", return_value=_SIGNALS), \
         patch("app.pipelines.technical.get_support_resistance", return_value=_SR):
        resp = client.get("/api/v1/technical/AAPL")

    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "AAPL"
    assert "signals" in data
    assert "RSI" in data["signals"]
    assert "latest_indicators" in data
    assert data["data_points"] == len(df)


def test_get_technical_overall_signal_buy_when_buy_majority(client: TestClient) -> None:
    """매수 신호가 과반이면 overall_signal이 BUY다."""
    df = _make_df()

    with patch("app.pipelines.fetcher.fetch_stock_data", return_value=(df, _INFO)), \
         patch("app.pipelines.technical.add_all_indicators", return_value=df), \
         patch("app.pipelines.technical.get_current_signals", return_value=_SIGNALS), \
         patch("app.pipelines.technical.get_support_resistance", return_value=_SR):
        resp = client.get("/api/v1/technical/AAPL")

    # _SIGNALS has BUY:2, SELL:1 → overall = BUY
    assert resp.json()["overall_signal"] == "BUY"


# ── 데이터 없음 ───────────────────────────────────────────────

def test_get_technical_no_data_returns_404(client: TestClient) -> None:
    """fetch_stock_data가 None을 반환하면 404를 반환한다."""
    with patch("app.pipelines.fetcher.fetch_stock_data", return_value=(None, None)):
        resp = client.get("/api/v1/technical/INVALID")

    assert resp.status_code == 404


# ── period_days 쿼리 파라미터 ─────────────────────────────────

def test_get_technical_period_days_passed_to_fetcher(client: TestClient) -> None:
    """period_days 쿼리 파라미터가 fetch_stock_data에 그대로 전달된다."""
    df = _make_df()

    with patch("app.pipelines.fetcher.fetch_stock_data", return_value=(df, _INFO)) as mock_fetch, \
         patch("app.pipelines.technical.add_all_indicators", return_value=df), \
         patch("app.pipelines.technical.get_current_signals", return_value=_SIGNALS), \
         patch("app.pipelines.technical.get_support_resistance", return_value=_SR):
        client.get("/api/v1/technical/AAPL?period_days=200")

    _, call_kwargs = mock_fetch.call_args
    assert call_kwargs.get("period_days") == 200 or mock_fetch.call_args[0][1] == 200


# ── 예상치 못한 예외 ──────────────────────────────────────────

def test_get_technical_unexpected_exception_returns_500(client: TestClient) -> None:
    """내부 예외 발생 시 500을 반환한다."""
    with patch("app.pipelines.fetcher.fetch_stock_data", side_effect=RuntimeError("unexpected")):
        resp = client.get("/api/v1/technical/AAPL")

    assert resp.status_code == 500


import pandas as pd  # noqa: E402  (환경 패치 후 import — 파일 상단 이동 불가)

# ─────────────────────────────────────────────────────────────
# 추가 엣지케이스 테스트
# ─────────────────────────────────────────────────────────────


def test_get_technical_no_data_returns_404_v2(client: TestClient) -> None:
    """fetch_stock_data None 반환 시 404 (추가 확인)."""
    with patch("app.pipelines.fetcher.fetch_stock_data", return_value=(None, None)):
        resp = client.get("/api/v1/technical/TOTALLY_INVALID_TICKER")
    assert resp.status_code == 404


def test_get_technical_runtime_error_returns_500_v2(client: TestClient) -> None:
    """내부 RuntimeError 시 500 (추가 확인)."""
    with patch("app.pipelines.fetcher.fetch_stock_data",
               side_effect=RuntimeError("unexpected error")):
        resp = client.get("/api/v1/technical/AAPL")
    assert resp.status_code == 500