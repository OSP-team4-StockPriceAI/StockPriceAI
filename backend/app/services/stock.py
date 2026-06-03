import json
import logging

import httpx
import redis.asyncio as redis
from fastapi import HTTPException, status

from app.core.config import settings
from app.core.redis import ml_redis_client
from app.schemas.stock import Candle, StockInfo

logger = logging.getLogger("stockai.backend.services.stock")

# 커넥션 풀 재사용을 위한 글로벌 지연(Lazy) 초기화 클라이언트 변수
_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """이벤트 루프 내부에서 안전하게 실행되도록 HTTPX AsyncClient를 지연 초기화하여 반환합니다."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=10.0)
    return _http_client


def get_ml_redis_client() -> redis.Redis:
    """ML 서비스의 Redis DB 1용 클라이언트를 반환합니다. (core/redis 모듈로 단일화)"""
    return ml_redis_client


async def fetch_stock_info(ticker: str) -> StockInfo:
    """ML 서비스에서 종목 정보를 조회하여 StockInfo로 반환합니다.

    [캐시 우선 조회 패턴 - Cache-First Policy]
    1. ML 서비스가 사용하는 Redis 캐시(DB 1)를 먼저 확인합니다.
    2. 캐시가 이미 존재하면 ML 서비스 호출(HTTP)을 완전히 생략하고 즉시 응답하여 초고속 반환합니다 (지연시간 ~2ms).
    3. 캐시가 존재하지 않는 경우에만 ML 서비스를 호출하여 실시간 수집 및 Redis 저장을 유도하고, 
       새롭게 저장된 캐시 데이터를 다시 조회하여 최종 결합 반환합니다.
    """
    ticker_upper = ticker.upper().strip()
    cache_payload = {}
    
    # 캐시 키 정의
    normalized_ticker = ticker_upper
    if ticker_upper.isdigit() and len(ticker_upper) == 6:
        normalized_ticker = f"{ticker_upper}.KS"
    cache_key = f"fetcher:stock:{normalized_ticker}"

    r = get_ml_redis_client()
    
    # 1. 1차 관문: ML Redis 캐시(DB 1)에서 데이터 선조회
    info = {}
    cached_price = None
    cache_found = False
    
    try:
        cached_data = await r.get(cache_key)
        if cached_data:
            cache_payload = json.loads(cached_data)
            info = cache_payload.get("info", {})
            
            # 캐시 내 history에서 가장 최신 종가를 추출하여 실시간 가격 대용으로 활용
            history = cache_payload.get("history", [])
            if history:
                cached_price = history[-1].get("Close")
            
            cache_found = True
            logger.info(f"⚡ [{ticker_upper}] ML Redis 캐시 적중! ML 서비스 API 호출을 생략하고 캐시 데이터를 즉시 반환합니다.")
    except Exception as e:
        logger.error(f"❌ [{ticker_upper}] ML Redis 캐시 1차 조회 오류: {str(e)}")

    # 2. 캐시가 적중되었고 기본 정보가 온전하다면 바로 반환 (네트워크 호출 제거로 성능 극대화)
    if cache_found and info:
        candles = []
        history = cache_payload.get("history", [])
        for c in history:
            if isinstance(c, dict):
                candles.append(
                    Candle(
                        Date=c.get("Date"),
                        Open=c.get("Open"),
                        High=c.get("High"),
                        Low=c.get("Low"),
                        Close=c.get("Close"),
                        Volume=c.get("Volume"),
                    )
                )
        return StockInfo(
            ticker=ticker_upper,
            name=info.get("longName") or info.get("shortName") or ticker_upper,
            sector=info.get("sector") or "Unknown",
            industry=info.get("industry") or "Unknown",
            market_cap=info.get("marketCap") or 0.0,
            current_price=cached_price,
            currency=info.get("currency") or "USD",
            history=candles,
        )

    # 3. 2차 관문: 캐시 미적중 시, ML 서비스를 직접 호출하여 새로운 캐시 생성 유도
    logger.info(f"🔍 [{ticker_upper}] 캐시 미적중. ML 서비스를 호출하여 실시간 수집 및 Redis 캐싱을 트리거합니다.")
    client = get_http_client()
    indicators = {}
    try:
        resp = await client.get(
            f"{settings.ML_SERVICE_URL}/api/v1/technical/{ticker_upper}"
        )
        resp.raise_for_status()
        data = resp.json()
        indicators = data.get("latest_indicators", {})
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.warning(f"⚠️ ML 서비스에서 티커를 찾을 수 없음: {ticker_upper}")
            return StockInfo(
                ticker=ticker_upper,
                name=ticker_upper,
                current_price=None,
            )
        logger.error(f"❌ ML 서비스 HTTP 에러 ({e.response.status_code}): {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"ML service error: {e.response.text}",
        )
    except httpx.RequestError as e:
        logger.error(f"❌ ML 서비스 통신 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML service unavailable",
        )

    # 4. ML 서비스가 수집 완료 후 Redis에 저장했으므로, 새로 생성된 캐시(DB 1)를 재조회하여 회사 메타데이터 확보
    try:
        cached_data = await r.get(cache_key)
        if cached_data:
            cache_payload = json.loads(cached_data)
            info = cache_payload.get("info", {})
            logger.info(f"✅ [{ticker_upper}] ML 호출 후 생성된 Redis 캐시에서 기본 정보 조회 성공")
    except Exception as e:
        logger.error(f"❌ [{ticker_upper}] ML 호출 후 Redis 캐시 재조회 오류: {str(e)}")

    # 5. 최종 데이터 조합 반환 (모든 필드 매핑)
    candles = []
    history = cache_payload.get("history", []) if cache_payload else []
    for c in history:
        if isinstance(c, dict):
            candles.append(
                Candle(
                    Date=c.get("Date"),
                    Open=c.get("Open"),
                    High=c.get("High"),
                    Low=c.get("Low"),
                    Close=c.get("Close"),
                    Volume=c.get("Volume"),
                )
            )

    return StockInfo(
        ticker=ticker_upper,
        name=info.get("longName") or info.get("shortName") or ticker_upper,
        sector=info.get("sector") or "Unknown",
        industry=info.get("industry") or "Unknown",
        market_cap=info.get("marketCap") or 0.0,
        current_price=indicators.get("close"),
        currency=info.get("currency") or "USD",
        history=candles,
    )
