"""scan_tasks.py 단위 테스트 — Redis mock으로 외부 의존성 격리"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ── _save_progress ────────────────────────────────────────────

@patch("app.workers.scan_tasks._get_redis")
def test_save_progress_stores_json_in_redis(mock_get_redis) -> None:
    """_save_progress가 올바른 키/TTL/JSON으로 setex를 호출한다."""
    from app.workers.scan_tasks import PROGRESS_KEY_PREFIX, PROGRESS_TTL, _save_progress

    mock_r = MagicMock()
    mock_get_redis.return_value = mock_r

    _save_progress("job-1", {"status": "running", "done": 3})

    mock_r.setex.assert_called_once()
    key, ttl, raw = mock_r.setex.call_args[0]
    assert key == f"{PROGRESS_KEY_PREFIX}job-1"
    assert ttl == PROGRESS_TTL
    assert json.loads(raw)["status"] == "running"


@patch("app.workers.scan_tasks._get_redis")
def test_save_progress_silently_ignores_redis_error(mock_get_redis) -> None:
    """Redis 오류 발생 시 예외를 외부로 전파하지 않는다."""
    from app.workers.scan_tasks import _save_progress

    mock_get_redis.side_effect = Exception("Redis down")
    _save_progress("job-1", {"status": "running"})  # 예외 없이 통과


# ── get_scan_progress ─────────────────────────────────────────

@patch("app.workers.scan_tasks._get_redis")
def test_get_scan_progress_returns_parsed_dict(mock_get_redis) -> None:
    """Redis에 데이터가 있으면 파싱된 dict를 반환한다."""
    from app.workers.scan_tasks import get_scan_progress

    mock_r = MagicMock()
    mock_r.get.return_value = json.dumps({"status": "running", "done": 5, "total": 10})
    mock_get_redis.return_value = mock_r

    result = get_scan_progress("job-1")

    assert result is not None
    assert result["status"] == "running"
    assert result["done"] == 5


@patch("app.workers.scan_tasks._get_redis")
def test_get_scan_progress_returns_none_when_key_missing(mock_get_redis) -> None:
    """Redis에 키가 없으면 None을 반환한다."""
    from app.workers.scan_tasks import get_scan_progress

    mock_r = MagicMock()
    mock_r.get.return_value = None
    mock_get_redis.return_value = mock_r

    result = get_scan_progress("nonexistent")

    assert result is None


@patch("app.workers.scan_tasks._get_redis")
def test_get_scan_progress_returns_none_on_redis_error(mock_get_redis) -> None:
    """Redis 오류 시 None을 반환하고 예외를 전파하지 않는다."""
    from app.workers.scan_tasks import get_scan_progress

    mock_get_redis.side_effect = Exception("connection error")

    result = get_scan_progress("job-1")

    assert result is None


# ── run_scan_job (Celery 태스크) ──────────────────────────────

@patch("app.workers.scan_tasks._save_progress")
@patch("app.pipelines.scanner.run_sp500_scan")
def test_run_scan_job_completes_and_saves_final_state(mock_scan, mock_save) -> None:
    """스캔 성공 시 status=completed 최종 상태를 Redis에 저장한다."""
    from app.workers.celery_app import celery_app
    from app.workers.scan_tasks import run_scan_job

    mock_scan.return_value = (pd.DataFrame(), {})

    celery_app.conf.task_always_eager = True
    try:
        result = run_scan_job.apply(
            kwargs={"job_id": "job-ok", "tickers": ["AAPL", "MSFT"], "max_workers": 1}
        ).get()
    finally:
        celery_app.conf.task_always_eager = False

    assert result["status"] == "completed"
    assert result["job_id"] == "job-ok"
    # 최종 저장 호출 확인 (초기 + 완료 최소 2회)
    assert mock_save.call_count >= 2
    last_saved = mock_save.call_args_list[-1][0][1]
    assert last_saved["status"] == "completed"


@patch("app.workers.scan_tasks._save_progress")
@patch("app.pipelines.scanner.run_sp500_scan")
def test_run_scan_job_saves_failed_state_on_exception(mock_scan, mock_save) -> None:
    """스캔 중 예외 발생 시 status=failed 상태를 Redis에 저장하고 예외를 재발생시킨다."""
    from app.workers.celery_app import celery_app
    from app.workers.scan_tasks import run_scan_job

    mock_scan.side_effect = RuntimeError("scan crashed")

    celery_app.conf.task_always_eager = True
    try:
        with pytest.raises(RuntimeError, match="scan crashed"):
            run_scan_job.apply(
                kwargs={"job_id": "job-fail", "tickers": ["AAPL"], "max_workers": 1}
            ).get()
    finally:
        celery_app.conf.task_always_eager = False

    # failed 상태가 저장되었는지 확인
    saved_states = [call[0][1] for call in mock_save.call_args_list]
    assert any(s.get("status") == "failed" for s in saved_states)


# ── _get_redis ────────────────────────────────────────────────

@patch("app.workers.scan_tasks.redis.from_url")
def test_get_redis_calls_from_url(mock_from_url) -> None:
    """_get_redis가 settings.redis_url로 redis.from_url을 호출한다."""
    from app.workers.scan_tasks import _get_redis

    mock_from_url.return_value = MagicMock()
    result = _get_redis()
    mock_from_url.assert_called_once()
    assert result is mock_from_url.return_value


# ── run_scan_job non-empty result ─────────────────────────────

@patch("app.workers.scan_tasks._save_progress")
@patch("app.pipelines.scanner.run_sp500_scan")
def test_run_scan_job_returns_top_results_when_scan_has_data(mock_scan, mock_save) -> None:
    """스캔 결과가 있으면 top_results에 데이터가 담겨 반환된다."""
    from app.workers.celery_app import celery_app
    from app.workers.scan_tasks import run_scan_job

    scan_df = pd.DataFrame([{
        "ticker": "AAPL", "name": "Apple", "sector": "Tech",
        "composite_score": 0.9, "up_probability": 0.8,
        "estimated_upside": 0.15, "ml_signal": "BUY",
        "current_price": 190.0, "rsi": 55.0,
        "buy_signals": 3, "market_cap": 3e12,
    }])
    mock_scan.return_value = (scan_df, {})

    celery_app.conf.task_always_eager = True
    try:
        result = run_scan_job.apply(
            kwargs={"job_id": "job-data", "tickers": ["AAPL"], "max_workers": 1}
        ).get()
    finally:
        celery_app.conf.task_always_eager = False

    assert result["status"] == "completed"
    assert len(result["results"]) == 1
    assert result["results"][0]["ticker"] == "AAPL"


# ── on_progress callback ──────────────────────────────────────

@patch("app.workers.scan_tasks._save_progress")
@patch("app.pipelines.scanner.run_sp500_scan")
def test_run_scan_job_progress_callback_is_invoked(mock_scan, mock_save) -> None:
    """run_sp500_scan이 progress_callback을 호출하면 _save_progress가 running 상태로 기록된다."""
    from app.workers.celery_app import celery_app
    from app.workers.scan_tasks import run_scan_job

    def fake_scan(tickers, max_workers, force_refresh, period_days, progress, progress_callback):
        progress_callback({"done": 1, "total": 1, "pct": 100.0, "cached": 0,
                           "refreshed": 1, "failed": 0, "current_ticker": "AAPL",
                           "elapsed_sec": 0.1, "eta_sec": None})
        return (pd.DataFrame(), {})

    mock_scan.side_effect = fake_scan

    celery_app.conf.task_always_eager = True
    try:
        result = run_scan_job.apply(
            kwargs={"job_id": "job-cb", "tickers": ["AAPL"], "max_workers": 1}
        ).get()
    finally:
        celery_app.conf.task_always_eager = False

    assert result["status"] == "completed"
    running_calls = [c for c in mock_save.call_args_list if c[0][1].get("status") == "running"]
    assert len(running_calls) >= 2  # 초기 + callback 호출


# ─────────────────────────────────────────────────────────────
# 추가 단위 테스트 (redis mock)
# ─────────────────────────────────────────────────────────────

class TestScanTasks:
    def test_get_scan_progress_returns_none_on_redis_failure(self):
        from app.workers.scan_tasks import get_scan_progress
        with patch("app.workers.scan_tasks._get_redis", side_effect=Exception("no redis")):
            result = get_scan_progress("test-job-id")
        assert result is None

    def test_get_scan_progress_returns_none_when_missing(self):
        from app.workers.scan_tasks import get_scan_progress
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        with patch("app.workers.scan_tasks._get_redis", return_value=mock_redis):
            result = get_scan_progress("nonexistent-job")
        assert result is None

    def test_get_scan_progress_returns_data_when_exists(self):
        from app.workers.scan_tasks import get_scan_progress
        data = {"job_id": "abc", "status": "running", "pct": 50.0}
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(data)
        with patch("app.workers.scan_tasks._get_redis", return_value=mock_redis):
            result = get_scan_progress("abc")
        assert result is not None
        assert result["status"] == "running"

    def test_save_progress_silently_fails_on_redis_error(self):
        from app.workers.scan_tasks import _save_progress
        with patch("app.workers.scan_tasks._get_redis", side_effect=Exception("no redis")):
            # 예외 없이 조용히 실패해야 함
            _save_progress("test-job", {"status": "running"})

    def test_save_progress_stores_data(self):
        from app.workers.scan_tasks import _save_progress
        mock_redis = MagicMock()
        with patch("app.workers.scan_tasks._get_redis", return_value=mock_redis):
            _save_progress("job123", {"status": "done"})
        mock_redis.setex.assert_called_once()

    def test_get_redis_creates_connection(self):
        from app.workers.scan_tasks import _get_redis
        with patch("redis.from_url", return_value=MagicMock()) as mock_redis:
            _get_redis()
        mock_redis.assert_called_once()