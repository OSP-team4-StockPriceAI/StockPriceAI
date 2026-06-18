from pydantic import BaseModel, Field


class SentimentSaveRequest(BaseModel):
    """단일 날짜 감정지수."""
    ticker: str = Field(..., description="종목 코드 (예: AAPL)")
    date: str = Field(..., description="YYYY-MM-DD (UTC)")
    avg_sentiment: float = Field(..., description="impact-weighted 종합 감성 점수 (-1~1)")
    time_weighted_avg: float | None = None
    raw_avg: float | None = None
    impact_score_avg: float | None = None
    news_count: int | None = None
    positive_pct: float | None = None
    negative_pct: float | None = None
    neutral_pct: float | None = None
    signal: str | None = Field(None, description="BULLISH / BEARISH / NEUTRAL")


class SentimentBatchSaveRequest(BaseModel):
    """ML 서비스 → Backend: 날짜별 감정지수 배치 저장 요청."""
    ticker: str = Field(..., description="종목 코드")
    items: list[SentimentSaveRequest] = Field(..., description="날짜별 감정지수 리스트")


class SentimentBatchSaveResponse(BaseModel):
    ticker: str
    saved: int    # 새로 저장된 건수
    skipped: int  # 이미 존재해서 스킵된 건수


class SentimentSaveResponse(BaseModel):
    ticker: str
    date: str
    avg_sentiment: float
    signal: str | None


class SentimentHistoryItem(BaseModel):
    date: str
    avg_sentiment: float
    time_weighted_avg: float | None = None
    raw_avg: float | None = None
    impact_score_avg: float | None = None
    news_count: int | None = None
    positive_pct: float | None = None
    negative_pct: float | None = None
    neutral_pct: float | None = None
    signal: str | None = None

    model_config = {"from_attributes": True}


class SentimentHistoryResponse(BaseModel):
    ticker: str
    count: int
    history: list[SentimentHistoryItem]


class SentimentPurgeResponse(BaseModel):
    ticker: str
    deleted: int
    cutoff_date: str