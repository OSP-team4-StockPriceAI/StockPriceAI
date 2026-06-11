import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class StockSentiment(Base):
    __tablename__ = "stock_sentiments"
    __table_args__ = (
        Index("idx_stock_sentiment_ticker_date", "ticker", "date"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    date: Mapped[str] = mapped_column(String(10), nullable=False)
    avg_sentiment: Mapped[float] = mapped_column(Float, nullable=False)
    time_weighted_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    impact_score_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    news_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    positive_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    negative_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    neutral_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
