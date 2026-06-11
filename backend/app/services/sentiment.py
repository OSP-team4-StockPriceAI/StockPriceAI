"""감정지수 DB 저장 / 조회 / 만료 삭제 서비스."""

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from app.models.sentiment import StockSentiment
from app.schemas.sentiment import (
    SentimentBatchSaveRequest,
    SentimentBatchSaveResponse,
    SentimentHistoryItem,
    SentimentSaveRequest,
    SentimentSaveResponse,
)

log = logging.getLogger("stockai.backend.sentiment")


# ─────────────────────────────────────────────────────────────
# 단일 저장 (upsert — 기존 호환용)
# ─────────────────────────────────────────────────────────────

def upsert_sentiment(db: Session, req: SentimentSaveRequest) -> StockSentiment:
    """ticker + date 기준 upsert."""
    existing = (
        db.query(StockSentiment)
        .filter(
            StockSentiment.ticker == req.ticker.upper(),
            StockSentiment.date == req.date,
        )
        .first()
    )
    if existing:
        existing.avg_sentiment    = req.avg_sentiment
        existing.time_weighted_avg = req.time_weighted_avg
        existing.raw_avg          = req.raw_avg
        existing.impact_score_avg = req.impact_score_avg
        existing.news_count       = req.news_count
        existing.positive_pct     = req.positive_pct
        existing.negative_pct     = req.negative_pct
        existing.neutral_pct      = req.neutral_pct
        existing.signal           = req.signal
        existing.updated_at       = datetime.now(UTC)
        db.commit()
        db.refresh(existing)
        return existing

    row = StockSentiment(
        ticker=req.ticker.upper(),
        date=req.date,
        avg_sentiment=req.avg_sentiment,
        time_weighted_avg=req.time_weighted_avg,
        raw_avg=req.raw_avg,
        impact_score_avg=req.impact_score_avg,
        news_count=req.news_count,
        positive_pct=req.positive_pct,
        negative_pct=req.negative_pct,
        neutral_pct=req.neutral_pct,
        signal=req.signal,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# ─────────────────────────────────────────────────────────────
# 배치 저장 — 이미 있는 날짜는 skip
# ─────────────────────────────────────────────────────────────

def batch_insert_sentiment(
    db: Session,
    req: SentimentBatchSaveRequest,
) -> SentimentBatchSaveResponse:
    """날짜별 감정지수 배치 저장. 이미 DB에 있는 날짜는 skip.

    ML 서비스가 30개 뉴스를 날짜별로 그룹핑한 결과를 한 번에 전송합니다.
    """
    ticker = req.ticker.upper()

    # 요청에 포함된 날짜 집합
    req_dates = {item.date for item in req.items}

    # DB에서 이미 존재하는 날짜를 한 번에 조회
    existing_dates: set[str] = {
        row.date
        for row in db.query(StockSentiment.date)
        .filter(
            StockSentiment.ticker == ticker,
            StockSentiment.date.in_(req_dates),
        )
        .all()
    }

    saved = 0
    skipped = 0
    new_rows: list[StockSentiment] = []

    for item in req.items:
        if item.date in existing_dates:
            skipped += 1
            continue
        new_rows.append(
            StockSentiment(
                ticker=ticker,
                date=item.date,
                avg_sentiment=item.avg_sentiment,
                time_weighted_avg=item.time_weighted_avg,
                raw_avg=item.raw_avg,
                impact_score_avg=item.impact_score_avg,
                news_count=item.news_count,
                positive_pct=item.positive_pct,
                negative_pct=item.negative_pct,
                neutral_pct=item.neutral_pct,
                signal=item.signal,
            )
        )
        saved += 1

    if new_rows:
        db.add_all(new_rows)
        db.commit()
        log.info(f"감정지수 배치 저장: {ticker} — 저장 {saved}건, 스킵 {skipped}건")

    return SentimentBatchSaveResponse(ticker=ticker, saved=saved, skipped=skipped)


# ─────────────────────────────────────────────────────────────
# 이력 조회
# ─────────────────────────────────────────────────────────────

def get_sentiment_history(
    db: Session,
    ticker: str,
    limit: int = 365,
) -> list[SentimentHistoryItem]:
    """날짜 오름차순(과거→최신)으로 반환."""
    rows = (
        db.query(StockSentiment)
        .filter(StockSentiment.ticker == ticker.upper())
        .order_by(StockSentiment.date.asc())
        .limit(limit)
        .all()
    )
    return [
        SentimentHistoryItem(
            date=r.date,
            avg_sentiment=r.avg_sentiment,
            time_weighted_avg=r.time_weighted_avg,
            raw_avg=r.raw_avg,
            impact_score_avg=r.impact_score_avg,
            news_count=r.news_count,
            positive_pct=r.positive_pct,
            negative_pct=r.negative_pct,
            neutral_pct=r.neutral_pct,
            signal=r.signal,
        )
        for r in rows
    ]


# ─────────────────────────────────────────────────────────────
# 만료 삭제 — 학습 기간(period_days) 초과 레코드 제거
# ─────────────────────────────────────────────────────────────

def purge_old_sentiments(
    db: Session,
    ticker: str,
    period_days: int,
) -> int:
    """cutoff(오늘 - period_days) 이전 레코드를 삭제하고 삭제 건수 반환."""
    cutoff = (datetime.now(UTC) - timedelta(days=period_days)).strftime("%Y-%m-%d")
    deleted = (
        db.query(StockSentiment)
        .filter(
            StockSentiment.ticker == ticker.upper(),
            StockSentiment.date < cutoff,
        )
        .delete(synchronize_session=False)
    )
    db.commit()
    if deleted:
        log.info(f"만료 감정지수 삭제: {ticker} — {deleted}건 (cutoff: {cutoff})")
    return int(deleted)