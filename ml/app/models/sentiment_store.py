"""감정지수 ↔ Backend DB 연동 유틸.

[저장]  save_sentiment_batch_to_backend()
        - analyze_news_sentiment()가 반환한 news_df를 날짜별로 그룹핑
        - 날짜별 avg_sentiment를 Backend POST /api/v1/sentiment/batch 로 전송
        - 이미 DB에 있는 날짜는 backend가 skip 처리

[만료 삭제]  purge_old_sentiments()
        - 학습 완료 후 period_days 초과 레코드를 Backend에 DELETE 요청

[로드]  load_sentiment_history() / merge_sentiment_into_df()
        - Backend GET /api/v1/sentiment/{ticker}/history 에서 과거 시계열 수신
        - OHLCV DataFrame에 date 기준 left-merge하여 피처 컬럼 자동 채움
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import numpy as np
import pandas as pd

from ..core.config import settings

log = logging.getLogger("stockai.ml.sentiment_store")

_BATCH_URL   = "{base}/api/v1/sentiment/batch"
_HISTORY_URL = "{base}/api/v1/sentiment/{ticker}/history"
_PURGE_URL   = "{base}/api/v1/sentiment/{ticker}/purge"


# ─────────────────────────────────────────────────────────────
# 내부 유틸: news_df → 날짜별 그룹핑
# ─────────────────────────────────────────────────────────────

def _group_by_date(
    news_df: pd.DataFrame,
    ticker: str,
) -> list[dict[str, Any]]:
    """news_df를 published_at 날짜 기준으로 그룹핑하여 날짜별 감정지수 리스트 반환."""
    if news_df.empty:
        return []

    df = news_df.copy()
    df["_date"] = pd.to_datetime(df["published_at"]).dt.strftime("%Y-%m-%d")

    items = []
    for date_str, group in df.groupby("_date"):
        # 시간 감쇠 가중 평균 (hours_ago 기준)
        time_w = np.exp(-group["hours_ago"].fillna(36) / 36)
        time_w_norm = time_w / time_w.sum() if time_w.sum() > 0 else time_w
        time_avg = float((group["compound"] * time_w_norm).sum())

        # impact_score 가중 평균
        impact_w = group["impact_score"].abs().clip(0.01, 3.0)
        combined = time_w * impact_w
        total_w = combined.sum()
        if total_w > 1e-9:
            direction = np.sign(group["impact_score"])
            magnitude = (group["impact_score"].abs() * combined).sum() / total_w
            sign_avg  = (direction * combined).sum() / total_w
            impact_avg = float(np.clip(sign_avg * magnitude * 0.5, -1.0, 1.0))
        else:
            impact_avg = time_avg

        avg_sent = impact_avg if abs(impact_avg) > 0.01 else time_avg
        raw_avg  = float(group["compound"].mean())
        imp_avg  = float(group["impact_score"].mean())
        total    = len(group)
        pos_pct  = round(int((group["label"] == "POSITIVE").sum()) / total * 100, 1)
        neg_pct  = round(int((group["label"] == "NEGATIVE").sum()) / total * 100, 1)
        neu_pct  = round(int((group["label"] == "NEUTRAL").sum())  / total * 100, 1)

        if avg_sent > 0.10:
            signal = "BULLISH"
        elif avg_sent < -0.10:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        items.append({
            "ticker":           ticker.upper(),
            "date":             str(date_str),
            "avg_sentiment":    round(avg_sent, 4),
            "time_weighted_avg": round(time_avg, 4),
            "raw_avg":          round(raw_avg, 4),
            "impact_score_avg": round(imp_avg, 4),
            "news_count":       total,
            "positive_pct":     pos_pct,
            "negative_pct":     neg_pct,
            "neutral_pct":      neu_pct,
            "signal":           signal,
        })

    items.sort(key=lambda x: str(x["date"]))
    return items


# ─────────────────────────────────────────────────────────────
# 저장: 배치 (동기 / 비동기)
# ─────────────────────────────────────────────────────────────

def save_sentiment_to_backend(
    ticker: str,
    news_df: pd.DataFrame,
    timeout: float = 5.0,
) -> dict[str, int]:
    """news_df를 날짜별 그룹핑 후 Backend에 동기 배치 저장.

    Returns: {"saved": N, "skipped": M}
    """
    items = _group_by_date(news_df, ticker)
    if not items:
        return {"saved": 0, "skipped": 0}

    url = _BATCH_URL.format(base=settings.backend_url)
    payload = {"ticker": ticker.upper(), "items": items}
    try:
        resp = httpx.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        log.info(
            f"감정지수 배치 저장: {ticker} — "
            f"저장 {data.get('saved', 0)}건, 스킵 {data.get('skipped', 0)}건"
        )
        return {"saved": data.get("saved", 0), "skipped": data.get("skipped", 0)}
    except Exception as e:
        log.warning(f"감정지수 배치 저장 실패 (비중요): {ticker} — {e}")
        return {"saved": 0, "skipped": 0}


async def save_sentiment_to_backend_async(
    ticker: str,
    news_df: pd.DataFrame,
    timeout: float = 5.0,
) -> dict[str, int]:
    """news_df를 날짜별 그룹핑 후 Backend에 비동기 배치 저장."""
    items = _group_by_date(news_df, ticker)
    if not items:
        return {"saved": 0, "skipped": 0}

    url = _BATCH_URL.format(base=settings.backend_url)
    payload = {"ticker": ticker.upper(), "items": items}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        log.info(
            f"감정지수 배치 저장(async): {ticker} — "
            f"저장 {data.get('saved', 0)}건, 스킵 {data.get('skipped', 0)}건"
        )
        return {"saved": data.get("saved", 0), "skipped": data.get("skipped", 0)}
    except Exception as e:
        log.warning(f"감정지수 배치 저장 실패(async, 비중요): {ticker} — {e}")
        return {"saved": 0, "skipped": 0}


# ─────────────────────────────────────────────────────────────
# 만료 삭제
# ─────────────────────────────────────────────────────────────

def purge_old_sentiments(
    ticker: str,
    period_days: int,
    timeout: float = 5.0,
) -> int:
    """학습 기간(period_days) 초과 레코드를 Backend에 삭제 요청."""
    url = _PURGE_URL.format(base=settings.backend_url, ticker=ticker.upper())
    try:
        resp = httpx.delete(url, params={"period_days": period_days}, timeout=timeout)
        resp.raise_for_status()
        deleted = int(resp.json().get("deleted", 0))
        if deleted:
            log.info(f"만료 감정지수 삭제: {ticker} — {deleted}건")
        return deleted
    except Exception as e:
        log.warning(f"만료 감정지수 삭제 실패 (비중요): {ticker} — {e}")
        return 0


# ─────────────────────────────────────────────────────────────
# 로드 / merge
# ─────────────────────────────────────────────────────────────

def load_sentiment_history(
    ticker: str,
    limit: int = 365,
    timeout: float = 5.0,
) -> pd.DataFrame:
    """Backend에서 과거 감정지수 시계열을 DataFrame으로 로드."""
    url = _HISTORY_URL.format(base=settings.backend_url, ticker=ticker.upper())
    try:
        resp = httpx.get(url, params={"limit": limit}, timeout=timeout)
        resp.raise_for_status()
        history = resp.json().get("history", [])
        if not history:
            return pd.DataFrame()

        df = pd.DataFrame(history)
        df["Sentiment_Score"]    = df["avg_sentiment"]
        df["Sentiment_Positive"] = df["avg_sentiment"].clip(lower=0)
        df["Sentiment_Negative"] = (-df["avg_sentiment"]).clip(lower=0)
        return df
    except Exception as e:
        log.warning(f"감정지수 이력 로드 실패 (비중요): {ticker} — {e}")
        return pd.DataFrame()


def merge_sentiment_into_df(
    df: pd.DataFrame,
    ticker: str,
    limit: int = 365,
) -> pd.DataFrame:
    """OHLCV + 기술지표 DataFrame에 과거 감정지수를 date 기준 left-merge.

    감정지수 없는 날짜(뉴스 없던 날)는 0으로 채워 학습에 지장 없도록 합니다.
    """
    sent_df = load_sentiment_history(ticker, limit=limit)

    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        # DatetimeIndex → "Date" 컬럼으로 추출 (reset_index 사용 시 index.name이
        # None이면 컬럼명이 "index"가 되어 KeyError 발생하므로 직접 할당)
        df["Date"] = df.index.strftime("%Y-%m-%d")
    elif df.index.name == "Date":
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    elif "Date" not in df.columns:
        log.warning("merge_sentiment_into_df: Date 컬럼/DatetimeIndex 없음 — 감정지수 0으로 채움")
        for col in ["Sentiment_Score", "Sentiment_Positive", "Sentiment_Negative"]:
            df[col] = 0.0
        df["Sentiment_Missing"] = 1.0
        return df
    else:
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    if sent_df.empty:
        for col in ["Sentiment_Score", "Sentiment_Positive", "Sentiment_Negative"]:
            df[col] = 0.0
        df["Sentiment_Missing"] = 1.0
        return df

    sent_cols = ["date", "Sentiment_Score", "Sentiment_Positive", "Sentiment_Negative"]
    merged = df.merge(
        sent_df[[c for c in sent_cols if c in sent_df.columns]].rename(columns={"date": "Date"}),
        on="Date",
        how="left",
    )
    # 뉴스가 없는 날을 명시적으로 표시 (0 = 데이터 있음, 1 = 데이터 없음)
    merged["Sentiment_Missing"] = merged["Sentiment_Score"].isna().astype("float32")

    for col in ["Sentiment_Score", "Sentiment_Positive", "Sentiment_Negative"]:
        merged[col] = merged.get(col, pd.Series(0.0, index=merged.index)).fillna(0.0)

    filled = merged["Sentiment_Score"].ne(0).sum()
    log.info(f"감정지수 merge 완료: {ticker} — {len(merged)}행 중 {filled}행 감정지수 있음")
    return merged