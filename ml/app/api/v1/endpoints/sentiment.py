"""GET /api/v1/sentiment/{ticker} — 감성 분석"""

import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ....models.sentiment import analyze_news_sentiment
from ....pipelines.fetcher import fetch_stock_data

router = APIRouter()
log = logging.getLogger("stockai.api.sentiment")


class NewsItem(BaseModel):
    title: str
    publisher: str
    hours_ago: float
    source: str
    compound: float
    label: str
    relevance: float
    relevance_tier: str
    news_type: str
    impact_score: float
    macro_theme: str | None = None


class SentimentResponse(BaseModel):
    ticker: str
    signal: str
    avg_sentiment: float
    time_weighted_avg: float
    raw_avg: float
    impact_score_avg: float
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    news_count: int
    direct_news_count: int
    surprise_count: int
    structural_count: int
    macro_themes: list[str]
    model: str
    sources: list[str]
    news: list[NewsItem]


@router.get("/{ticker}", response_model=SentimentResponse, summary="뉴스 감성 분석")
async def get_sentiment(
    ticker: str,
    max_news: int = Query(default=30, ge=5, le=100, description="최대 뉴스 수"),
    use_finbert: bool = Query(default=False, description="FinBERT 사용 (느리지만 정확)"),
) -> SentimentResponse:
    """
    종목 관련 뉴스를 수집하고 감성을 분석합니다.

    - yfinance → Yahoo Finance RSS → Google News RSS 순으로 폴백
    - VADER + 금융 특화 사전 기반 Impact Score 가중 감성 점수
    - **signal**: BULLISH / NEUTRAL / BEARISH
    """
    try:
        ticker = ticker.strip().upper()

        _, info = fetch_stock_data(ticker, period_days=90)
        company_name = ""
        sector = ""
        beta = 1.0
        if info:
            company_name = info.get("shortName", "")
            sector = info.get("sector", "")
            beta = float(info.get("beta", 1.0) or 1.0)

        news_df, summary = analyze_news_sentiment(
            ticker=ticker,
            company_name=company_name,
            sector=sector,
            use_finbert=use_finbert,
            max_news=max_news,
            beta=beta,
        )

        news_items: list[NewsItem] = []
        if not news_df.empty:
            for _, row in news_df.head(max_news).iterrows():
                news_items.append(
                    NewsItem(
                        title=str(row.get("title", "")),
                        publisher=str(row.get("publisher", "")),
                        hours_ago=float(row.get("hours_ago", 0)),
                        source=str(row.get("source", "")),
                        compound=float(row.get("compound", 0)),
                        label=str(row.get("label", "NEUTRAL")),
                        relevance=float(row.get("relevance", 0)),
                        relevance_tier=str(row.get("relevance_tier", "")),
                        news_type=str(row.get("news_type", "")),
                        impact_score=float(row.get("impact_score", 0)),
                        macro_theme=row.get("macro_theme") or None,
                    )
                )

        return SentimentResponse(
            ticker=ticker,
            signal=summary["signal"],
            avg_sentiment=summary["avg_sentiment"],
            time_weighted_avg=summary["time_weighted_avg"],
            raw_avg=summary["raw_avg"],
            impact_score_avg=summary["impact_score_avg"],
            positive_pct=summary["positive_pct"],
            negative_pct=summary["negative_pct"],
            neutral_pct=summary["neutral_pct"],
            news_count=summary["news_count"],
            direct_news_count=summary["direct_news_count"],
            surprise_count=summary["surprise_count"],
            structural_count=summary["structural_count"],
            macro_themes=summary["macro_themes"],
            model=summary["model"],
            sources=summary["sources"],
            news=news_items,
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"감성 분석 실패: {ticker}")
        raise HTTPException(status_code=500, detail=str(e))