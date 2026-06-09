"""scanner 엔드포인트 테스트 (POST /start, GET /status, GET /tickers, GET /cache/stats)"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

_START_PAYLOAD = {"tickers": ["AAPL", "MSFT"], "max_workers": 1, "force_refresh": False, "period_days": 400}


# ── POST /api/v1/scanner/start ────────────────────────────────

def test_start_scan_returns_job_id(client: TestClient) -> None:
    """정상 요청 시 job_id와 queued 상태를 반환한다."""
    with patch("app.api.v1.endpoints.scanner._save_progress"), \
         patch("app.api.v1.endpoints.scanner.run_scan_job") as mock_task:
        mock_task.apply_async.return_value = None
        resp = client.post("/api/v1/scanner/start", json=_START_PAYLOAD)

    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    assert data["total"] == 2


def test_start_scan_saves_initial_progress_before_dispatch(client: TestClient) -> None:
    """apply_async 전에 _save_progress가 반드시 먼저 호출된다."""
    call_order = []

    def fake_save(job_id, data):
        call_order.append("save")

    mock_task = MagicMock()
    mock_task.apply_async.side_effect = lambda **_: call_order.append("async")

    with patch("app.api.v1.endpoints.scanner._save_progress", side_effect=fake_save), \
         patch("app.api.v1.endpoints.scanner.run_scan_job", mock_task):
        client.post("/api/v1/scanner/start", json=_START_PAYLOAD)

    assert call_order.index("save") < call_order.index("async")


def test_start_scan_empty_tickers_returns_400(client: TestClient) -> None:
    """tickers를 빈 리스트로 명시하면 400을 반환한다."""
    # SP500_TICKERS가 아닌 빈 리스트를 강제로 주려면 SP500_TICKERS mock 필요
    with patch("app.api.v1.endpoints.scanner.SP500_TICKERS", []), \
         patch("app.api.v1.endpoints.scanner._save_progress"), \
         patch("app.api.v1.endpoints.scanner.run_scan_job"):
        resp = client.post("/api/v1/scanner/start", json={"tickers": []})

    assert resp.status_code == 400


def test_start_scan_celery_error_returns_503(client: TestClient) -> None:
    """Celery/Redis 연결 실패 시 503을 반환한다."""
    with patch("app.api.v1.endpoints.scanner._save_progress"), \
         patch("app.api.v1.endpoints.scanner.run_scan_job") as mock_task:
        mock_task.apply_async.side_effect = Exception("Redis connection refused")
        resp = client.post("/api/v1/scanner/start", json=_START_PAYLOAD)

    assert resp.status_code == 503


# ── GET /api/v1/scanner/status/{job_id} ──────────────────────

def test_get_scan_status_returns_progress(client: TestClient) -> None:
    """Redis에 진행 데이터가 있으면 200과 상태를 반환한다."""
    progress = {
        "job_id": "job-abc", "status": "running",
        "total": 10, "done": 4, "pct": 40.0,
        "cached": 2, "refreshed": 2, "failed": 0, "current_ticker": "GOOG",
    }

    with patch("app.api.v1.endpoints.scanner.get_scan_progress", return_value=progress):
        resp = client.get("/api/v1/scanner/status/job-abc")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert data["done"] == 4
    assert data["pct"] == 40.0


def test_get_scan_status_not_found_returns_404(client: TestClient) -> None:
    """Redis에 데이터가 없으면 404를 반환한다."""
    with patch("app.api.v1.endpoints.scanner.get_scan_progress", return_value=None):
        resp = client.get("/api/v1/scanner/status/nonexistent-job")

    assert resp.status_code == 404


# ── GET /api/v1/scanner/tickers ───────────────────────────────

def test_list_tickers_returns_list(client: TestClient) -> None:
    """SP500 종목 코드 목록을 반환한다."""
    resp = client.get("/api/v1/scanner/tickers")

    assert resp.status_code == 200
    tickers = resp.json()
    assert isinstance(tickers, list)
    assert len(tickers) > 0
    assert "AAPL" in tickers


# ── GET /api/v1/scanner/cache/stats ──────────────────────────

def test_cache_stats_returns_stats(client: TestClient) -> None:
    """캐시 통계를 정상 반환한다."""
    stats = {"valid": 10, "stale": 2, "uncached": 3, "total_cached": 12, "ttl_hours": 24}

    with patch("app.api.v1.endpoints.scanner.get_cache_stats", return_value=stats):
        resp = client.get("/api/v1/scanner/cache/stats?limit=15")

    assert resp.status_code == 200
    assert resp.json()["valid"] == 10


def test_cache_stats_exception_returns_500(client: TestClient) -> None:
    """get_cache_stats가 예외를 던지면 500을 반환한다."""
    with patch("app.api.v1.endpoints.scanner.get_cache_stats", side_effect=Exception("Redis down")):
        resp = client.get("/api/v1/scanner/cache/stats")

    assert resp.status_code == 500
