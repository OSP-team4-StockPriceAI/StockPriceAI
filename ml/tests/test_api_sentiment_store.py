"""GET /api/v1/sentiment/{ticker} 엔드포인트 테스트

네트워크 없이 실행되도록 analyze_news_sentiment / fetch_stock_data를 전부 mock 처리.
"""

from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ─────────────────────────────────────────────────────────────
# 픽스처 / 헬퍼
# ─────────────────────────────────────────────────────────────

_INFO = {
    "shortName": "Apple Inc.",
    "sector": "Technology",
    "beta": 1.2,
}

_SUMMARY = {
    "signal": "BULLISH",
    "avg_sentiment": 0.35,
    "time_weighted_avg": 0.32,
    "raw_avg": 0.30,
    "impact_score_avg": 0.40,
    "positive_pct": 60.0,
    "negative_pct": 20.0,
    "neutral_pct": 20.0,
    "news_count": 10,
    "direct_news_count": 7,
    "surprise_count": 2,
    "structural_count": 1,
    "macro_themes": ["FED", "INFLATION"],
    "model": "vader",
    "sources": ["Yahoo Finance", "Google News"],
}


def _make_news_df(n: int = 5) -> pd.DataFrame:
    """analyze_news_sentiment()가 반환하는 news_df 형태 더미."""
    return pd.DataFrame({
        "title":          [f"News {i}" for i in range(n)],
        "publisher":      ["Reuters"] * n,
        "hours_ago":      [float(i) for i in range(n)],
        "source":         ["yahoo"] * n,
        "compound":       [0.3] * n,
        "label":          ["POSITIVE"] * n,
        "relevance":      [0.9] * n,
        "relevance_tier": ["HIGH"] * n,
        "news_type":      ["direct"] * n,
        "impact_score":   [1.2] * n,
        "macro_theme":    [None] * n,
        "published_at":   pd.date_range("2024-01-01", periods=n, freq="h"),
    })


def _patch_all(news_df=None, summary=None, info=None, fetch_df=None):
    """analyze_news_sentiment + fetch_stock_data를 한 번에 패치하는 컨텍스트."""
    if news_df is None:
        news_df = _make_news_df()
    if summary is None:
        summary = _SUMMARY.copy()
    if info is None:
        info = _INFO.copy()

    return (
        patch(
            "app.api.v1.endpoints.sentiment.fetch_stock_data",
            return_value=(fetch_df, info),
        ),
        patch(
            "app.api.v1.endpoints.sentiment.analyze_news_sentiment",
            return_value=(news_df, summary),
        ),
    )


# ─────────────────────────────────────────────────────────────
# 정상 응답
# ─────────────────────────────────────────────────────────────

def test_get_sentiment_success_returns_200(client: TestClient) -> None:
    """정상 요청 시 SentimentResponse 스키마에 맞는 200 응답을 반환한다."""
    p1, p2 = _patch_all()
    with p1, p2:
        resp = client.get("/api/v1/sentiment/AAPL")

    assert resp.status_code == 200


def test_get_sentiment_ticker_uppercased(client: TestClient) -> None:
    """소문자 ticker도 대문자로 정규화되어 응답에 포함된다."""
    p1, p2 = _patch_all()
    with p1, p2:
        resp = client.get("/api/v1/sentiment/aapl")

    assert resp.json()["ticker"] == "AAPL"


def test_get_sentiment_response_schema(client: TestClient) -> None:
    """응답 JSON이 SentimentResponse 필드를 모두 포함한다."""
    p1, p2 = _patch_all()
    with p1, p2:
        resp = client.get("/api/v1/sentiment/AAPL")

    data = resp.json()
    required_fields = {
        "ticker", "signal", "avg_sentiment", "time_weighted_avg",
        "raw_avg", "impact_score_avg", "positive_pct", "negative_pct",
        "neutral_pct", "news_count", "direct_news_count", "surprise_count",
        "structural_count", "macro_themes", "model", "sources", "news",
    }
    assert required_fields <= set(data.keys())


def test_get_sentiment_summary_values_reflected(client: TestClient) -> None:
    """summary 딕셔너리 값이 응답에 그대로 반영된다."""
    p1, p2 = _patch_all()
    with p1, p2:
        resp = client.get("/api/v1/sentiment/AAPL")

    data = resp.json()
    assert data["signal"] == "BULLISH"
    assert data["avg_sentiment"] == pytest.approx(0.35)
    assert data["news_count"] == 10
    assert data["macro_themes"] == ["FED", "INFLATION"]
    assert "Yahoo Finance" in data["sources"]


def test_get_sentiment_news_items_in_response(client: TestClient) -> None:
    """news_df 행이 NewsItem 리스트로 변환되어 응답에 포함된다."""
    p1, p2 = _patch_all(news_df=_make_news_df(3))
    with p1, p2:
        resp = client.get("/api/v1/sentiment/AAPL")

    news = resp.json()["news"]
    assert len(news) == 3
    assert news[0]["title"] == "News 0"
    assert news[0]["label"] == "POSITIVE"
    assert news[0]["compound"] == pytest.approx(0.3)


def test_get_sentiment_news_item_schema(client: TestClient) -> None:
    """NewsItem이 필수 필드를 모두 포함한다."""
    p1, p2 = _patch_all(news_df=_make_news_df(1))
    with p1, p2:
        resp = client.get("/api/v1/sentiment/AAPL")

    item = resp.json()["news"][0]
    required = {
        "title", "publisher", "hours_ago", "source",
        "compound", "label", "relevance", "relevance_tier",
        "news_type", "impact_score",
    }
    assert required <= set(item.keys())


def test_get_sentiment_macro_theme_none_allowed(client: TestClient) -> None:
    """macro_theme이 None인 뉴스도 오류 없이 직렬화된다."""
    news_df = _make_news_df(1)
    news_df["macro_theme"] = None
    p1, p2 = _patch_all(news_df=news_df)
    with p1, p2:
        resp = client.get("/api/v1/sentiment/AAPL")

    assert resp.status_code == 200
    assert resp.json()["news"][0]["macro_theme"] is None


def test_get_sentiment_empty_news_df_returns_empty_list(client: TestClient) -> None:
    """뉴스가 없어도(news_df 비어있음) 200을 반환하고 news는 빈 리스트다."""
    p1, p2 = _patch_all(news_df=pd.DataFrame())
    with p1, p2:
        resp = client.get("/api/v1/sentiment/AAPL")

    assert resp.status_code == 200
    assert resp.json()["news"] == []


# ─────────────────────────────────────────────────────────────
# fetch_stock_data 결과별 분기
# ─────────────────────────────────────────────────────────────

def test_get_sentiment_info_none_uses_empty_defaults(client: TestClient) -> None:
    """fetch_stock_data가 info=None을 반환해도 감성 분석은 실행된다."""
    p1, p2 = _patch_all(info=None)
    with p1, p2:
        resp = client.get("/api/v1/sentiment/AAPL")

    assert resp.status_code == 200


def test_get_sentiment_passes_company_name_and_sector(client: TestClient) -> None:
    """fetch_stock_data의 info에서 추출한 company_name, sector가 analyze_news_sentiment에 전달된다."""
    p1, p2 = _patch_all()
    with p1, p2 as mock_analyze:
        client.get("/api/v1/sentiment/AAPL")

    call_kwargs = mock_analyze.call_args.kwargs
    assert call_kwargs.get("company_name") == "Apple Inc."
    assert call_kwargs.get("sector") == "Technology"


def test_get_sentiment_passes_beta_from_info(client: TestClient) -> None:
    """info의 beta 값이 analyze_news_sentiment에 float으로 전달된다."""
    info = {**_INFO, "beta": 1.5}
    p1, p2 = _patch_all(info=info)
    with p1, p2 as mock_analyze:
        client.get("/api/v1/sentiment/AAPL")

    assert mock_analyze.call_args.kwargs.get("beta") == pytest.approx(1.5)


def test_get_sentiment_beta_none_defaults_to_1(client: TestClient) -> None:
    """info의 beta가 None이면 1.0으로 대체되어 전달된다."""
    info = {**_INFO, "beta": None}
    p1, p2 = _patch_all(info=info)
    with p1, p2 as mock_analyze:
        client.get("/api/v1/sentiment/AAPL")

    assert mock_analyze.call_args.kwargs.get("beta") == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────
# 쿼리 파라미터
# ─────────────────────────────────────────────────────────────

def test_get_sentiment_max_news_limits_news_list(client: TestClient) -> None:
    """max_news=5이면 news 리스트가 최대 5개로 제한된다."""
    p1, p2 = _patch_all(news_df=_make_news_df(10))
    with p1, p2:
        resp = client.get("/api/v1/sentiment/AAPL?max_news=5")

    assert len(resp.json()["news"]) == 5


def test_get_sentiment_max_news_passed_to_analyze(client: TestClient) -> None:
    """max_news 파라미터가 analyze_news_sentiment에 전달된다."""
    p1, p2 = _patch_all()
    with p1, p2 as mock_analyze:
        client.get("/api/v1/sentiment/AAPL?max_news=15")

    assert mock_analyze.call_args.kwargs.get("max_news") == 15


def test_get_sentiment_use_finbert_passed_to_analyze(client: TestClient) -> None:
    """use_finbert=true 파라미터가 analyze_news_sentiment에 전달된다."""
    p1, p2 = _patch_all()
    with p1, p2 as mock_analyze:
        client.get("/api/v1/sentiment/AAPL?use_finbert=true")

    assert mock_analyze.call_args.kwargs.get("use_finbert") is True


def test_get_sentiment_use_finbert_default_false(client: TestClient) -> None:
    """use_finbert 기본값은 False다."""
    p1, p2 = _patch_all()
    with p1, p2 as mock_analyze:
        client.get("/api/v1/sentiment/AAPL")

    assert mock_analyze.call_args.kwargs.get("use_finbert") is False


def test_get_sentiment_max_news_below_min_returns_422(client: TestClient) -> None:
    """max_news < 5이면 FastAPI 유효성 검사 오류 422를 반환한다."""
    resp = client.get("/api/v1/sentiment/AAPL?max_news=2")
    assert resp.status_code == 422


def test_get_sentiment_max_news_above_max_returns_422(client: TestClient) -> None:
    """max_news > 100이면 FastAPI 유효성 검사 오류 422를 반환한다."""
    resp = client.get("/api/v1/sentiment/AAPL?max_news=101")
    assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────
# 에러 처리
# ─────────────────────────────────────────────────────────────

def test_get_sentiment_analyze_exception_returns_500(client: TestClient) -> None:
    """analyze_news_sentiment에서 예외 발생 시 500을 반환한다."""
    with patch(
        "app.api.v1.endpoints.sentiment.fetch_stock_data",
        return_value=(None, _INFO),
    ), patch(
        "app.api.v1.endpoints.sentiment.analyze_news_sentiment",
        side_effect=RuntimeError("model load failed"),
    ):
        resp = client.get("/api/v1/sentiment/AAPL")

    assert resp.status_code == 500
    assert "model load failed" in resp.json()["detail"]


def test_get_sentiment_fetch_exception_returns_500(client: TestClient) -> None:
    """fetch_stock_data에서 예외 발생 시 500을 반환한다."""
    with patch(
        "app.api.v1.endpoints.sentiment.fetch_stock_data",
        side_effect=ConnectionError("yfinance timeout"),
    ):
        resp = client.get("/api/v1/sentiment/AAPL")

    assert resp.status_code == 500


# ─────────────────────────────────────────────────────────────
# signal 값 검증
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("signal", ["BULLISH", "BEARISH", "NEUTRAL"])
def test_get_sentiment_all_signals_accepted(client: TestClient, signal: str) -> None:
    """BULLISH / BEARISH / NEUTRAL 세 가지 signal 모두 응답에 그대로 포함된다."""
    summary = {**_SUMMARY, "signal": signal}
    p1, p2 = _patch_all(summary=summary)
    with p1, p2:
        resp = client.get("/api/v1/sentiment/AAPL")

    assert resp.json()["signal"] == signal