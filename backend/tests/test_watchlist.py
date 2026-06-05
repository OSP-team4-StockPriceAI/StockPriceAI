from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


def test_list_watchlist_empty(client: TestClient, auth_headers: dict) -> None:
    resp = client.get("/api/v1/watchlist", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []


def test_add_and_list_watchlist(client: TestClient, auth_headers: dict) -> None:
    resp = client.post(
        "/api/v1/watchlist",
        json={"ticker": "aapl", "memo": "Apple stock"},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["ticker"] == "AAPL"
    assert data["memo"] == "Apple stock"

    resp = client.get("/api/v1/watchlist", headers=auth_headers)
    assert len(resp.json()) == 1


def test_add_duplicate_ticker(client: TestClient, auth_headers: dict) -> None:
    client.post("/api/v1/watchlist", json={"ticker": "MSFT"}, headers=auth_headers)
    resp = client.post("/api/v1/watchlist", json={"ticker": "msft"}, headers=auth_headers)
    assert resp.status_code == 409


def test_update_watchlist_memo(client: TestClient, auth_headers: dict) -> None:
    client.post("/api/v1/watchlist", json={"ticker": "GOOG"}, headers=auth_headers)
    resp = client.patch("/api/v1/watchlist/GOOG", json={"memo": "Updated memo"}, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["memo"] == "Updated memo"


def test_delete_watchlist_item(client: TestClient, auth_headers: dict) -> None:
    client.post("/api/v1/watchlist", json={"ticker": "NVDA"}, headers=auth_headers)
    resp = client.delete("/api/v1/watchlist/NVDA", headers=auth_headers)
    assert resp.status_code == 204

    resp = client.get("/api/v1/watchlist", headers=auth_headers)
    assert resp.json() == []


def test_delete_nonexistent_ticker(client: TestClient, auth_headers: dict) -> None:
    resp = client.delete("/api/v1/watchlist/FAKE", headers=auth_headers)
    assert resp.status_code == 404


def test_watchlist_requires_auth(client: TestClient) -> None:
    resp = client.get("/api/v1/watchlist")
    assert resp.status_code == 403


@patch("app.api.v1.endpoints.watchlist.get_ml_redis_client")
def test_list_watchlist_with_cache(mock_get_redis, client: TestClient, auth_headers: dict) -> None:
    # 1. Mock Redis client and its pipeline
    mock_redis = MagicMock()
    mock_pipe = MagicMock()
    mock_get_redis.return_value = mock_redis
    mock_redis.pipeline.return_value = mock_pipe
    
    # Mock pipe.execute() to return a mock JSON string
    mock_json = """{
        "info": {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "marketCap": 3000000000000.0,
            "currency": "USD"
        },
        "history": [
            {"Close": 185.5}
        ]
    }"""
    mock_pipe.execute = AsyncMock(return_value=[mock_json])
    
    # 2. Add an item to watchlist
    client.post("/api/v1/watchlist", json={"ticker": "AAPL", "memo": "Apple stock"}, headers=auth_headers)
    
    # 3. Get watchlist and check if new fields are populated
    resp = client.get("/api/v1/watchlist", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["ticker"] == "AAPL"
    assert data[0]["name"] == "Apple Inc."
    assert data[0]["sector"] == "Technology"
    assert data[0]["industry"] == "Consumer Electronics"
    assert data[0]["market_cap"] == 3000000000000.0
    assert data[0]["current_price"] == 185.5
    assert data[0]["currency"] == "USD"
