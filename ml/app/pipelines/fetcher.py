"""
Multi-source Data Ingestion Module
- yfinance: 실시간 시세 + 재무지표
- 한국/미국 주식 모두 지원
"""

import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import redis
import yfinance as yf

from ..core.config import settings

log = logging.getLogger("stockai.fetcher")

warnings.filterwarnings("ignore")


def is_korean_ticker(ticker: str) -> bool:
    """한국 종목 코드 판별 (숫자 6자리 또는 .KS/.KQ 접미사)"""
    ticker_clean = ticker.upper().strip()
    if ticker_clean.isdigit() and len(ticker_clean) == 6:
        return True
    if ticker_clean.endswith(".KS") or ticker_clean.endswith(".KQ"):
        return True
    return False


def normalize_ticker(ticker: str) -> str:
    """한국 종목 코드를 yfinance 형식으로 변환"""
    ticker = ticker.strip().upper()
    if ticker.isdigit() and len(ticker) == 6:
        return f"{ticker}.KS"
    return ticker


def fetch_stock_data(
    ticker: str,
    period_days: int = 365,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """
    주가 데이터 및 재무정보 수집
    1. ML 내부 Redis 캐시(settings.redis_url) 우선 확인
    2. 캐시가 없으면 yfinance로 수집 후 Redis에 저장(TTL 24시간)

    Returns:
        (price_df, financials_dict)
    """
    ticker = normalize_ticker(ticker)
    cache_key = f"fetcher:stock:{ticker}"
    r = None
    info: dict[str, Any] = {}

    # 1. 내부 Redis 캐시에서 먼저 조회 시도 (force_refresh가 False일 때만)
    if not force_refresh:
        try:
            r = redis.from_url(settings.redis_url, decode_responses=True)  # type: ignore[no-untyped-call]
            cached_data = r.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                df = pd.DataFrame(data.get("history", []))
                if not df.empty:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df.set_index("Date", inplace=True)
                    for col in ["Open", "High", "Low", "Close", "Volume"]:
                        if col in df.columns:
                            df[col] = df[col].astype("float32")
                    info = data.get("info", {})
                    log.info(f"✅ [{ticker}] 내부 Redis 캐시에서 데이터 로드 완료")
                    return df, info
        except Exception as e:
            log.warning(f"⚠️ [{ticker}] Redis 캐시 조회 실패: {e}. yfinance로 폴백합니다.")
    else:
        # force_refresh가 True이면 무조건 새로 수집하기 위해 r만 초기화
        try:
            r = redis.from_url(settings.redis_url, decode_responses=True)  # type: ignore[no-untyped-call]
        except Exception:
            pass

    # 2. 캐시 미스 또는 실패 시 yfinance 직접 호출 (폴백)
    log.info(f"🌐 [{ticker}] yfinance에서 데이터 수집 중...")
    try:
        stock = yf.Ticker(ticker)

        if period_days <= 0:
            hist = stock.history(period="max", auto_adjust=True)
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days + 365)
            hist = stock.history(start=start_date, end=end_date, auto_adjust=True)

        if hist.empty or len(hist) < 30:
            return None, None

        hist.index = pd.to_datetime(hist.index)
        if hist.index.tz is not None:
            hist.index = hist.index.tz_convert("UTC").tz_localize(None)
        hist = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
        hist.dropna(inplace=True)

        for col in ["Open", "High", "Low", "Close"]:
            hist[col] = hist[col].astype("float32")
        hist["Volume"] = hist["Volume"].astype("float32")

        info = {}
        try:
            raw_info = stock.info
            financial_keys = [
                "trailingPE",
                "forwardPE",
                "priceToBook",
                "returnOnEquity",
                "returnOnAssets",
                "debtToEquity",
                "currentRatio",
                "quickRatio",
                "revenueGrowth",
                "earningsGrowth",
                "grossMargins",
                "operatingMargins",
                "profitMargins",
                "marketCap",
                "enterpriseValue",
                "dividendYield",
                "beta",
                "fiftyTwoWeekHigh",
                "fiftyTwoWeekLow",
                "shortName",
                "longName",
                "sector",
                "industry",
                "currency",
                "regularMarketPrice",
                "regularMarketChangePercent",
                "averageVolume",
                "sharesOutstanding",
            ]
            for key in financial_keys:
                val = raw_info.get(key)
                if val is not None:
                    info[key] = val
        except Exception:
            pass

        # 3. yfinance 수집 성공 시 Redis에 저장 (TTL 24시간)
        if r is not None and not hist.empty:
            try:
                hist_reset = hist.copy()
                hist_reset.index.name = "Date"
                hist_reset.reset_index(inplace=True)
                hist_reset["Date"] = hist_reset["Date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
                cache_payload = {
                    "info": info,
                    "history": hist_reset.to_dict(orient="records")
                }
                # 하루(86400초) 캐시
                r.setex(cache_key, 86400, json.dumps(cache_payload, default=str))
                log.info(f"💾 [{ticker}] 새로운 데이터를 Redis 캐시에 저장했습니다. (TTL: 24h)")
            except Exception as e:
                log.warning(f"⚠️ [{ticker}] Redis 캐시 저장 실패: {e}")

        return hist, info

    except Exception as e:
        raise ValueError(f"데이터 수집 실패 ({ticker}): {str(e)}")


def fetch_earnings_history(ticker: str) -> pd.DataFrame | None:
    """분기별 실적 데이터 수집"""
    ticker = normalize_ticker(ticker)
    try:
        stock = yf.Ticker(ticker)

        try:
            stmt = stock.quarterly_income_stmt
            if stmt is not None and not stmt.empty:
                df = stmt.T.sort_index(ascending=False)
                cols = [
                    c
                    for c in df.columns
                    if any(
                        k in str(c).lower()
                        for k in ["revenue", "net income", "gross profit", "ebit"]
                    )
                ]
                if cols:
                    return df[cols].head(8)
                return df.head(8)
        except Exception:
            pass

        try:
            qf = stock.quarterly_financials
            if qf is not None and not qf.empty:
                return qf.T.sort_index(ascending=False).head(8)
        except Exception:
            pass

    except Exception:
        pass
    return None


def fetch_institutional_holders(ticker: str) -> pd.DataFrame | None:
    """기관 투자자 보유 현황"""
    ticker = normalize_ticker(ticker)
    try:
        stock = yf.Ticker(ticker)
        holders = stock.institutional_holders
        if holders is not None and not holders.empty:
            return holders.head(10)
    except Exception:
        pass
    return None


def get_market_context(ticker: str) -> dict[str, Any]:
    """시장 맥락 데이터 (벤치마크 대비 성과)"""
    context: dict[str, Any] = {}

    norm_ticker = normalize_ticker(ticker)
    if ".KS" in norm_ticker or ".KQ" in norm_ticker:
        benchmark = "^KS11"
        benchmark_name = "KOSPI"
    else:
        benchmark = "^GSPC"
        benchmark_name = "S&P 500"

    try:
        bench_data = yf.Ticker(benchmark).history(period="3mo")
        if not bench_data.empty:
            bench_return = (bench_data["Close"].iloc[-1] / bench_data["Close"].iloc[0] - 1) * 100
            context["benchmark_name"] = benchmark_name
            context["benchmark_3mo_return"] = round(bench_return, 2)
    except Exception:
        pass

    return context
