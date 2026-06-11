"""GET /api/v1/sentiment/{ticker} 엔드포인트 테스트"""

from datetime import datetime
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient


def _make_summary(news_count: int = 3) -> dict:
    return {
        "signal": "BULLISH",
        "avg_sentiment": 0.35,
        "time_weighted_avg": 0.40,
        "raw_avg": 0.30,
        "impact_score_avg": 0.25,
        "positive_pct": 70.0,
        "negative_pct": 10.0,
        "neutral_pct": 20.0,
        "news_count": news_count,
        "direct_news_count": 2,
        "surprise_count": 1,
        "structural_count": 0,
        "macro_themes": [],
        "model": "VADER + Impact Framework",
        "sources": ["yfinance"],
    }


def _make_news_df(n: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "title": f"News headline {i}",
            "publisher": "Reuters",
            "published_at": datetime.now(),
            "url": "https://example.com",
            "hours_ago": float(i + 1),
            "source": "yfinance",
            "compound": 0.4,
            "label": "POSITIVE",
            "positive": 0.6, "negative": 0.1, "neutral": 0.3,
            "relevance": 0.8,
            "relevance_tier": "직접",
            "news_type": "general",
            "persistence": 0.3,
            "contagion": 0.0,
            "macro_theme": None,
            "macro_exposure": 0.0,
            "market_regime": 1.2,
            "impact_score": 0.25,
            "S_surprise": 0.4,
            "M_regime": 1.2,
            "P_persistence": 0.3,
        })
    return pd.DataFrame(rows)


# ── 정상 감성 분석 ────────────────────────────────────────────

def test_get_sentiment_success_returns_200(client: TestClient) -> None:
    """정상 요청 시 SentimentResponse 스키마에 맞는 200 응답을 반환한다."""
    info = {"shortName": "Apple", "sector": "Tech", "beta": 1.2}
    with patch("app.pipelines.fetcher.fetch_stock_data",
               return_value=(None, info)), \
         patch("app.models.sentiment.analyze_news_sentiment",
               return_value=(_make_news_df(2), _make_summary(2))):
        resp = client.get("/api/v1/sentiment/AAPL")

    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "AAPL"
    assert data["signal"] == "BULLISH"
    assert data["news_count"] == 2
    assert len(data["news"]) == 2
    assert data["news"][0]["title"] == "News headline 0"


# ── 뉴스 없음 ─────────────────────────────────────────────────

def test_get_sentiment_no_news_returns_empty_list(client: TestClient) -> None:
    """뉴스가 없으면 news=[]/signal=NEUTRAL인 200을 반환한다."""
    empty_summary = {**_make_summary(0), "signal": "NEUTRAL", "model": "N/A",
                     "avg_sentiment": 0.0, "time_weighted_avg": 0.0,
                     "raw_avg": 0.0, "impact_score_avg": 0.0,
                     "positive_pct": 0.0, "negative_pct": 0.0, "neutral_pct": 0.0}

    with patch("app.pipelines.fetcher.fetch_stock_data", return_value=(None, {})), \
         patch("app.models.sentiment.analyze_news_sentiment",
               return_value=(pd.DataFrame(), empty_summary)):
        resp = client.get("/api/v1/sentiment/AAPL")

    assert resp.status_code == 200
    assert resp.json()["news"] == []
    assert resp.json()["signal"] == "NEUTRAL"


# ── use_finbert 파라미터 전달 ─────────────────────────────────

def test_get_sentiment_use_finbert_param_passed(client: TestClient) -> None:
    """use_finbert=true 파라미터가 analyze_news_sentiment에 전달된다."""
    with patch("app.pipelines.fetcher.fetch_stock_data", return_value=(None, {})), \
         patch("app.models.sentiment.analyze_news_sentiment",
               return_value=(pd.DataFrame(), _make_summary(0))) as mock_analyze:
        client.get("/api/v1/sentiment/AAPL?use_finbert=true")

    call_kwargs = mock_analyze.call_args[1]
    assert call_kwargs.get("use_finbert") is True


# ── 예상치 못한 예외 ──────────────────────────────────────────

def test_get_sentiment_unexpected_exception_returns_500(client: TestClient) -> None:
    """내부 예외 발생 시 500을 반환한다."""
    with patch("app.pipelines.fetcher.fetch_stock_data", side_effect=RuntimeError("boom")):
        resp = client.get("/api/v1/sentiment/AAPL")

    assert resp.status_code == 500