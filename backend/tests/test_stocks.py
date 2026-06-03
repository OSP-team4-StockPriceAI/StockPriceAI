import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient


def _cache_payload(*, with_info: bool = True, price: float = 185.5) -> str:
    payload: dict = {}
    if with_info:
        payload["info"] = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "marketCap": 3_000_000_000_000.0,
            "currency": "USD",
        }
    payload["history"] = [{"Close": price}]
    return json.dumps(payload)


def _mock_redis(cached: str | None) -> AsyncMock:
    r = AsyncMock()
    r.get.return_value = cached
    return r


# ── Redis 캐시 히트 ───────────────────────────────────────────

@patch("app.services.stock.get_ml_redis_client")
def test_get_stock_cache_hit_returns_stockinfo(mock_get_redis, client: TestClient) -> None:
    """캐시 적중 시 ML 호출 없이 즉시 StockInfo를 반환한다."""
    mock_get_redis.return_value = _mock_redis(_cache_payload())

    with patch("app.services.stock.get_http_client") as mock_http:
        resp = client.get("/api/v1/stocks/AAPL")

    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "AAPL"
    assert data["name"] == "Apple Inc."
    assert data["sector"] == "Technology"
    assert data["current_price"] == 185.5
    mock_http.assert_not_called()


@patch("app.services.stock.get_ml_redis_client")
def test_get_stock_ticker_uppercased(mock_get_redis, client: TestClient) -> None:
    """소문자 ticker도 대문자로 정규화된다."""
    mock_get_redis.return_value = _mock_redis(_cache_payload())

    resp = client.get("/api/v1/stocks/aapl")

    assert resp.status_code == 200
    assert resp.json()["ticker"] == "AAPL"


# ── Redis 캐시 미스 → ML 호출 ─────────────────────────────────

@patch("app.services.stock.get_http_client")
@patch("app.services.stock.get_ml_redis_client")
def test_get_stock_cache_miss_calls_ml_technical(mock_get_redis, mock_get_http, client: TestClient) -> None:
    """캐시 미적중 시 ML 기술적 지표 API를 호출하고, 재조회 후 결합 반환한다."""
    mock_redis = AsyncMock()
    # 1차 조회: 미스, 2차 조회(ML 호출 후): 적중
    mock_redis.get.side_effect = [None, _cache_payload()]
    mock_get_redis.return_value = mock_redis

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"latest_indicators": {"close": 190.0}}
    mock_http_client = AsyncMock()
    mock_http_client.get.return_value = mock_resp
    mock_get_http.return_value = mock_http_client

    resp = client.get("/api/v1/stocks/AAPL")

    assert resp.status_code == 200
    mock_http_client.get.assert_called_once()


# ── ML 서비스 오류 ────────────────────────────────────────────

@patch("app.services.stock.get_http_client")
@patch("app.services.stock.get_ml_redis_client")
def test_get_stock_ml_404_returns_empty_stockinfo(mock_get_redis, mock_get_http, client: TestClient) -> None:
    """ML이 404를 반환하면 200 + current_price=None인 StockInfo를 반환한다."""
    mock_get_redis.return_value = _mock_redis(None)

    mock_raw_resp = MagicMock()
    mock_raw_resp.status_code = 404
    mock_raw_resp.text = "not found"
    mock_http_client = AsyncMock()
    mock_http_client.get.return_value = mock_raw_resp
    mock_raw_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404", request=MagicMock(), response=mock_raw_resp
    )
    mock_get_http.return_value = mock_http_client

    resp = client.get("/api/v1/stocks/AAPL")

    assert resp.status_code == 200
    assert resp.json()["ticker"] == "AAPL"
    assert resp.json()["current_price"] is None


@patch("app.services.stock.get_http_client")
@patch("app.services.stock.get_ml_redis_client")
def test_get_stock_ml_offline_returns_503(mock_get_redis, mock_get_http, client: TestClient) -> None:
    """ML 서비스 연결 실패 시 503을 반환한다."""
    mock_get_redis.return_value = _mock_redis(None)

    mock_http_client = AsyncMock()
    mock_http_client.get.side_effect = httpx.RequestError("connection refused")
    mock_get_http.return_value = mock_http_client

    resp = client.get("/api/v1/stocks/AAPL")

    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"].lower()


# ── 인증 없이 접근 가능 ───────────────────────────────────────

@patch("app.services.stock.get_ml_redis_client")
def test_get_stock_requires_no_auth(mock_get_redis, client: TestClient) -> None:
    """/stocks 엔드포인트는 JWT 없이도 접근 가능하다."""
    mock_get_redis.return_value = _mock_redis(_cache_payload())

    resp = client.get("/api/v1/stocks/MSFT")

    assert resp.status_code == 200
