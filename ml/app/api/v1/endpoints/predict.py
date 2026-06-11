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
    include_sentiment: bool = Field(default=True, description="감성 분석 포함 여부")
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
    try:
        from ....models.predictor import EnsemblePredictor
        from ....models.sentiment import analyze_news_sentiment
        from ....models.sentiment_store import (
            merge_sentiment_into_df,
            purge_old_sentiments,
            save_sentiment_to_backend_async,
        )
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
        df = label_training_target(df)

        if req.include_sentiment:
            # 1. 뉴스 분석 — news_df(개별 기사) + summary(요약) 반환
            news_df, sent_summary = analyze_news_sentiment(
                ticker=ticker,
                company_name=info.get("shortName", "") if info else "",
                sector=info.get("sector", "") if info else "",
            )

            # 2. news_df를 날짜별로 그룹핑해서 Backend DB에 배치 저장 (중복 날짜 skip)
            await save_sentiment_to_backend_async(ticker, news_df)

            # 3. 학습 기간 초과 레코드 삭제
            purge_old_sentiments(ticker, req.period_days)

            # 4. DB 이력을 학습 df에 merge (LSTM 피처용)
            df = merge_sentiment_into_df(df, ticker, limit=req.period_days)

            # 5. 오늘 행 감정지수 보정 (방금 저장됐지만 DB 응답 전일 수 있으므로)
            today_score = sent_summary.get("avg_sentiment", 0.0)
            for col, val in [
                ("Sentiment_Score",    today_score),
                ("Sentiment_Positive", max(0.0, today_score)),
                ("Sentiment_Negative", max(0.0, -today_score)),
            ]:
                if col not in df.columns:
                    df[col] = 0.0
                if df[col].iloc[-1] == 0.0:
                    df.loc[df.index[-1], col] = val

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
                "rsi14":        
                float(df["RSI14"].iloc[-1])        if "RSI14"        in df else None,
                "bb_position":  
                float(df["BB_Position"].iloc[-1])  if "BB_Position"  in df else None,
                "volume_ratio": 
                float(df["Volume_Ratio"].iloc[-1]) if "Volume_Ratio" in df else None,
                "macd":         
                float(df["MACD"].iloc[-1])         if "MACD"         in df else None,
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