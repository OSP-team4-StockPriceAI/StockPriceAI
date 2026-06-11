"""감정지수 저장 / 조회 / 만료 삭제 엔드포인트.

POST /api/v1/sentiment/save          — 단일 날짜 upsert (기존 호환)
POST /api/v1/sentiment/batch         — 날짜별 배치 저장 (중복 날짜 skip)
GET  /api/v1/sentiment/{ticker}/history  — 이력 조회 (ML 학습용)
DELETE /api/v1/sentiment/{ticker}/purge  — 학습 기간 초과 레코드 삭제
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.sentiment import (
    SentimentBatchSaveRequest,
    SentimentBatchSaveResponse,
    SentimentHistoryResponse,
    SentimentPurgeResponse,
    SentimentSaveRequest,
    SentimentSaveResponse,
)
from app.services.sentiment import (
    batch_insert_sentiment,
    get_sentiment_history,
    purge_old_sentiments,
    upsert_sentiment,
)

router = APIRouter(prefix="/sentiment", tags=["Sentiment"])


@router.post("/save", response_model=SentimentSaveResponse, summary="단일 감정지수 저장")
def save_sentiment(
    req: SentimentSaveRequest,
    db: Session = Depends(get_db),
) -> SentimentSaveResponse:
    row = upsert_sentiment(db, req)
    return SentimentSaveResponse(
        ticker=row.ticker,
        date=row.date,
        avg_sentiment=row.avg_sentiment,
        signal=row.signal,
    )


@router.post("/batch", response_model=SentimentBatchSaveResponse, summary="날짜별 감정지수 배치 저장")
def save_sentiment_batch(
    req: SentimentBatchSaveRequest,
    db: Session = Depends(get_db),
) -> SentimentBatchSaveResponse:
    """뉴스 30개를 날짜별로 그룹핑한 감정지수를 한 번에 저장합니다.
    
    이미 DB에 존재하는 날짜는 덮어쓰지 않고 skip합니다.
    """
    return batch_insert_sentiment(db, req)


@router.get(
    "/{ticker}/history",
    response_model=SentimentHistoryResponse,
    summary="감정지수 이력 조회 (ML 학습용)",
)
def get_history(
    ticker: str,
    limit: int = Query(default=365, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> SentimentHistoryResponse:
    history = get_sentiment_history(db, ticker, limit=limit)
    return SentimentHistoryResponse(
        ticker=ticker.upper(),
        count=len(history),
        history=history,
    )


@router.delete(
    "/{ticker}/purge",
    response_model=SentimentPurgeResponse,
    summary="학습 기간 초과 감정지수 삭제",
)
def purge_sentiment(
    ticker: str,
    period_days: int = Query(..., ge=1, description="학습 기간(일) — 이보다 오래된 레코드 삭제"),
    db: Session = Depends(get_db),
) -> SentimentPurgeResponse:
    """오늘 기준 period_days 이전 데이터를 삭제합니다.
    
    ML 서비스가 학습 완료 후 호출하여 불필요한 과거 데이터를 정리합니다.
    """
    from datetime import UTC, datetime, timedelta
    cutoff = (datetime.now(UTC) - timedelta(days=period_days)).strftime("%Y-%m-%d")
    deleted = purge_old_sentiments(db, ticker, period_days)
    return SentimentPurgeResponse(ticker=ticker.upper(), deleted=deleted, cutoff_date=cutoff)