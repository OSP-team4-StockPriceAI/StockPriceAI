import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.prediction import Prediction
from app.schemas.prediction import PredictionOut

log = logging.getLogger("stockai.backend.prediction")

# 커넥션 풀 재사용을 위한 글로벌 지연(Lazy) 초기화 HTTPX 클라이언트 변수
_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """이벤트 루프 내부에서 안전하게 실행되도록 HTTPX AsyncClient를 지연 초기화하여 반환합니다.
    ML 예측은 대형 모델 학습 및 연산이 길게 유지되므로 타임아웃을 120초로 넉넉하게 설정합니다.
    """
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=120.0)
    return _http_client


async def get_or_fetch_predictions(
    ticker: str,
    limit: int,
    db: Session,
) -> list[PredictionOut]:
    """DB에서 예측 이력을 조회하고, 없거나 만료(24시간 초과)된 경우 ML 서비스에 예측을 요청하여 저장 후 반환합니다.

    [장애 복원성 + 캐시 만료 정책 - Cache Expiry & Resilient Fallback]
    1. DB에 해당 티커의 가장 최근 예측 기록을 가져와 24시간 이내에 생성된 "신선한(Fresh)" 데이터인지 확인합니다.
    2. 신선한 데이터가 존재하면 ML 서비스 호출 없이 바로 최신 기록(최대 limit개)을 반환합니다. (캐시 히트 ⚡)
    3. 데이터가 아예 없거나, 가장 최신 기록이 24시간을 초과하여 만료된 경우 ML 서비스를 호출해 신규 예측을 갱신합니다.
    4. ML 서비스가 네트워크 장애/타임아웃으로 실패할 경우, 만료된 기존 과거 예측 기록(Stale Backup)이라도
       대체 우회 반환하여 503 크래시를 완벽히 방지합니다.
       단, ML 서비스가 명시적인 HTTP 오류(4xx/5xx)를 반환한 경우에는 그 상태코드를 그대로 전파합니다.
    """
    ticker_upper = ticker.upper().strip()

    # 1. DB에서 가장 최신 예측 기록 1건만 선조회하여 최신성(Freshness) 검증
    latest_row = (
        db.query(Prediction)
        .filter(Prediction.ticker == ticker_upper)
        .order_by(Prediction.created_at.desc())
        .first()
    )

    is_fresh = False
    if latest_row:
        created_at = latest_row.created_at
        # SQLite 등 테스트 환경에서 datetime에 시간대(tzinfo) 정보가 유실되는 경우를 대비해 보정합니다.
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)

        time_diff = datetime.now(UTC) - created_at
        if time_diff < timedelta(hours=24):
            is_fresh = True

    # 2. 24시간 이내에 생성된 신선한 캐시가 있으면 즉시 리턴 (네트워크 지연 생략)
    if is_fresh:
        rows = (
            db.query(Prediction)
            .filter(Prediction.ticker == ticker_upper)
            .order_by(Prediction.created_at.desc())
            .limit(limit)
            .all()
        )
        log.info(f"⚡ [{ticker_upper}] 24시간 이내의 신선한 DB 예측 캐시 적중! 즉시 리턴합니다.")
        return [PredictionOut.model_validate(r) for r in rows]

    # 3. 신선한 캐시가 없거나 만료된 경우 ML 서비스 호출하여 새로운 예측 생성
    log.info(f"🔮 [{ticker_upper}] 예측 캐시가 없거나 만료(24시간 초과)됨 → ML 서비스를 호출하여 실시간 분석을 수행합니다.")
    try:
        ml_result = await _call_ml_predict(ticker_upper)
    except HTTPException:
        # ML 서비스가 명시적 HTTP 오류(4xx/5xx)를 반환한 경우:
        # stale fallback 없이 그 상태코드를 클라이언트에 그대로 전달합니다.
        # (HTTPException은 Exception의 서브클래스이므로 아래 except에서 삼켜지지 않도록 먼저 처리)
        raise
    except Exception as e:
        # 네트워크 장애 / 타임아웃 등 연결 수준 오류 → Stale Fallback 시도
        if latest_row:
            log.warning(
                f"🔄 [{ticker_upper}] ML 서비스 장애 감지. 만료된 과거 예측 데이터(Stale Backup, 생성일: {latest_row.created_at})로 우회 반환합니다."
            )
            rows = (
                db.query(Prediction)
                .filter(Prediction.ticker == ticker_upper)
                .order_by(Prediction.created_at.desc())
                .limit(limit)
                .all()
            )
            return [PredictionOut.model_validate(r) for r in rows]

        # 과거 백업 데이터조차 전혀 없는 최악의 경우 503으로 응답
        log.error(f"❌ [{ticker_upper}] ML 서비스 장애 및 DB 내 과거 예측 백업본 부재: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML service unavailable and no cached predictions found",
        )

    # 4. ML 호출 성공 시 응답을 DB에 저장
    ensemble_detail = ml_result.get("ensemble_detail") or {}
    new_prediction = Prediction(
        ticker=ticker_upper,
        signal=ml_result["signal"],
        up_prob=ml_result["up_probability"],
        model_type=ml_result.get("model"),
        complexity=ensemble_detail.get("complexity"),
        xgb_weight=ensemble_detail.get("w_xgb"),
        lstm_weight=ensemble_detail.get("w_lstm"),
    )
    db.add(new_prediction)
    db.commit()
    db.refresh(new_prediction)

    log.info(f"✅ ML 예측 결과 DB 저장 완료 및 갱신 성공: {ticker_upper} → {new_prediction.signal}")

    # 5. 방금 추가된 신규 건을 포함하여 최신 정렬로 반환
    rows = (
        db.query(Prediction)
        .filter(Prediction.ticker == ticker_upper)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
        .all()
    )
    return [PredictionOut.model_validate(r) for r in rows]


async def _call_ml_predict(ticker: str) -> dict[str, Any]:
    """ML 서비스의 /api/v1/predict 엔드포인트를 호출하여 예측 결과를 반환합니다.
    글로벌 HTTPX 클라이언트 풀을 사용해 소켓 커넥션을 재사용합니다.
    """
    client = get_http_client()
    try:
        resp = await client.post(
            f"{settings.ML_SERVICE_URL}/api/v1/predict",
            json={"ticker": ticker.upper()},
        )
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"ML service error: {e.response.text}",
        )
    except httpx.RequestError as e:
        log.error(f"❌ ML 서비스 예측 API 호출 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML service unavailable",
        )