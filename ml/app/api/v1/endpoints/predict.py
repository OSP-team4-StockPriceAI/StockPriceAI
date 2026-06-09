"""POST /api/v1/predict — 단일 종목 예측"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()
log = logging.getLogger("stockai.api.predict")


class PredictRequest(BaseModel):
    ticker: str = Field(..., description="종목 코드 (예: AAPL, 005930.KS)")
    period_days: int = Field(default=400, ge=100, le=3000, description="학습 기간(일)")
    include_sentiment: bool = Field(default=False, description="감성 분석 포함 여부")
    force_lstm: bool = Field(default=True, description="LSTM 강제 사용")


class PredictResponse(BaseModel):
    ticker: str
    signal: str
    up_probability: float
    down_probability: float
    confidence: float
    model: str
    ensemble_detail: dict[str, Any] | None = None
    training_metrics: dict[str, Any]
    technical_summary: dict[str, Any]


@router.post("", response_model=PredictResponse, summary="단일 종목 ML 예측")
async def predict(req: PredictRequest) -> PredictResponse:
    """
    종목의 다음날 상승/하락 확률을 XGBoost + LSTM 앙상블로 예측합니다.

    - **signal**: BUY / HOLD / SELL
    - **up_probability**: 상승 확률 (0~1)
    - **confidence**: 신뢰도 (max(up, down) 확률)
    """
    try:
        from ....models.predictor import EnsemblePredictor
        from ....models.sentiment import add_sentiment_to_features, analyze_news_sentiment
        from ....pipelines.fetcher import fetch_stock_data
        from ....pipelines.technical import (
            add_all_indicators,
            get_current_signals,
            get_support_resistance,
            label_training_target,
        )

        ticker = req.ticker.strip().upper()

        df, info = fetch_stock_data(ticker, period_days=req.period_days)
        if df is None:
            raise HTTPException(status_code=404, detail=f"데이터 없음: {ticker}")

        df = add_all_indicators(df)
        df = label_training_target(df)  # Target 컬럼 생성 (학습에 필요)

        if req.include_sentiment:
            _, sent_summary = analyze_news_sentiment(
                ticker=ticker,
                company_name=info.get("shortName", "") if info else "",
                sector=info.get("sector", "") if info else "",
            )
            df = add_sentiment_to_features(df, sent_summary["avg_sentiment"])

        predictor = EnsemblePredictor(scanner_mode=False)
        train_metrics = predictor.train(
            df, include_sentiment=req.include_sentiment, force_lstm=req.force_lstm
        )

        if "error" in train_metrics:
            raise HTTPException(status_code=422, detail=train_metrics["error"])

        pred = predictor.predict(df)
        if "error" in pred:
            raise HTTPException(status_code=500, detail=pred["error"])

        signals = get_current_signals(df)
        sr = get_support_resistance(df)

        tech_summary = {
            "signals": {k: {"action": v[0], "description": v[1]} for k, v in signals.items()},
            "support_resistance": sr,
            "latest": {
                "rsi14": float(df["RSI14"].iloc[-1]) if "RSI14" in df else None,
                "bb_position": float(df["BB_Position"].iloc[-1]) if "BB_Position" in df else None,
                "volume_ratio": (
                    float(df["Volume_Ratio"].iloc[-1]) if "Volume_Ratio" in df else None
                ),
                "macd": float(df["MACD"].iloc[-1]) if "MACD" in df else None,
            },
        }

        return PredictResponse(
            ticker=ticker,
            signal=pred["signal"],
            up_probability=pred["up_probability"],
            down_probability=pred["down_probability"],
            confidence=pred["confidence"],
            model=pred["model"],
            ensemble_detail=pred.get("ensemble_detail"),
            training_metrics=train_metrics,
            technical_summary=tech_summary,
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"예측 실패: {req.ticker}")
        raise HTTPException(status_code=500, detail=str(e))
