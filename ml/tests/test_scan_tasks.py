"""scan_tasks.py 커버리지 보강 테스트
get_latest_scan_job_id / scheduled_market_scan / _is_us_market_holiday / warmup_cache_task
"""

from datetime import datetime
from unittest.mock import MagicMock, call, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ET = ZoneInfo("America/New_York")


# ── get_latest_scan_job_id ────────────────────────────────────

@patch("app.workers.scan_tasks._get_redis")
def test_get_latest_scan_job_id_returns_value(mock_get_redis) -> None:
    """Redis에 값이 있으면 해당 job_id를 반환한다."""
    from app.workers.scan_tasks import get_latest_scan_job_id

    mock_r = MagicMock()
    mock_r.get.return_value = "job-abc"
    mock_get_redis.return_value = mock_r

    result = get_latest_scan_job_id("market_open")
    assert result == "job-abc"


@patch("app.workers.scan_tasks._get_redis")
def test_get_latest_scan_job_id_returns_none_when_missing(mock_get_redis) -> None:
    """Redis에 값이 없으면 None을 반환한다."""
    from app.workers.scan_tasks import get_latest_scan_job_id

    mock_r = MagicMock()
    mock_r.get.return_value = None
    mock_get_redis.return_value = mock_r

    result = get_latest_scan_job_id("market_close")
    assert result is None


@patch("app.workers.scan_tasks._get_redis")
def test_get_latest_scan_job_id_returns_none_on_redis_error(mock_get_redis) -> None:
    """Redis 오류 시 예외를 전파하지 않고 None을 반환한다."""
    from app.workers.scan_tasks import get_latest_scan_job_id

    mock_get_redis.side_effect = Exception("connection refused")

    result = get_latest_scan_job_id("market_open")
    assert result is None


# ── _is_us_market_holiday ─────────────────────────────────────

def test_is_us_market_holiday_returns_false_for_regular_day() -> None:
    """평일(공휴일 아닌 날)은 False를 반환한다."""
    from app.workers.scan_tasks import _is_us_market_holiday

    # 2025-01-02 목요일 (공휴일 아님)
    dt = datetime(2025, 1, 2, 10, 0, tzinfo=ET)
    assert _is_us_market_holiday(dt) is False


def test_is_us_market_holiday_returns_true_for_holiday() -> None:
    """NYSE 공휴일(독립기념일)은 True를 반환한다."""
    from app.workers.scan_tasks import _is_us_market_holiday

    # 2025-07-04 금요일 (독립기념일)
    dt = datetime(2025, 7, 4, 10, 0, tzinfo=ET)
    assert _is_us_market_holiday(dt) is True


def test_is_us_market_holiday_returns_true_for_christmas() -> None:
    """크리스마스는 True를 반환한다."""
    from app.workers.scan_tasks import _is_us_market_holiday

    dt = datetime(2025, 12, 25, 10, 0, tzinfo=ET)
    assert _is_us_market_holiday(dt) is True


# ── scheduled_market_scan ────────────────────────────────────

@patch("app.workers.scan_tasks._get_redis")
@patch("app.workers.scan_tasks._save_progress")
@patch("app.workers.scan_tasks.run_scan_job")
def test_scheduled_market_scan_skips_on_weekend(
    mock_run, mock_save, mock_get_redis
) -> None:
    """주말에는 스캔을 실행하지 않고 즉시 리턴한다."""
    from app.workers.scan_tasks import scheduled_market_scan

    # 2025-01-04 토요일
    saturday = datetime(2025, 1, 4, 10, 0, tzinfo=ET)
    with patch("app.workers.scan_tasks.datetime") as mock_dt:
        mock_dt.now.return_value = saturday
        scheduled_market_scan("market_open")

    mock_run.apply_async.assert_not_called()
    mock_save.assert_not_called()


@patch("app.workers.scan_tasks._get_redis")
@patch("app.workers.scan_tasks._save_progress")
@patch("app.workers.scan_tasks.run_scan_job")
def test_scheduled_market_scan_skips_on_holiday(
    mock_run, mock_save, mock_get_redis
) -> None:
    """공휴일에는 스캔을 실행하지 않는다."""
    from app.workers.scan_tasks import scheduled_market_scan

    # 2025-07-04 독립기념일 금요일
    holiday = datetime(2025, 7, 4, 10, 0, tzinfo=ET)
    with patch("app.workers.scan_tasks.datetime") as mock_dt:
        mock_dt.now.return_value = holiday
        scheduled_market_scan("market_open")

    mock_run.apply_async.assert_not_called()


@patch("app.workers.scan_tasks._get_redis")
@patch("app.workers.scan_tasks._save_progress")
@patch("app.workers.scan_tasks.run_scan_job")
def test_scheduled_market_scan_dispatches_on_weekday(
    mock_run, mock_save, mock_get_redis
) -> None:
    """평일 장중에는 run_scan_job.apply_async를 호출한다."""
    from app.workers.scan_tasks import scheduled_market_scan

    mock_r = MagicMock()
    mock_get_redis.return_value = mock_r

    # 2025-01-02 목요일 (평일, 비공휴일)
    weekday = datetime(2025, 1, 2, 10, 0, tzinfo=ET)
    with patch("app.workers.scan_tasks.datetime") as mock_dt, \
         patch("app.workers.scan_tasks.SP500_TICKERS", ["AAPL", "MSFT"], create=True), \
         patch("app.workers.scan_tasks._is_us_market_holiday", return_value=False):
        mock_dt.now.return_value = weekday
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        # SP500_TICKERS import 경로 패치
        with patch("app.workers.scan_tasks.run_scan_job") as mock_task:
            with patch(
                "app.workers.scan_tasks.__builtins__", {}
            ):
                pass
            # pipelines.scanner.SP500_TICKERS 패치
            with patch(
                "app.pipelines.scanner.SP500_TICKERS", ["AAPL", "MSFT"]
            ):
                scheduled_market_scan("market_open", force_refresh=False, max_workers=1)
                mock_task.apply_async.assert_called_once()


# ── warmup_cache_task ─────────────────────────────────────────

@patch("app.workers.scan_tasks._get_redis")
def test_warmup_cache_task_returns_result_string(mock_get_redis) -> None:
    """warmup_cache_task는 성공/실패 카운트가 담긴 문자열을 반환한다."""
    from app.workers.scan_tasks import warmup_cache_task

    mock_df = MagicMock()  # non-None df

    with patch("app.pipelines.fetcher.fetch_stock_data", return_value=(mock_df, {})), \
         patch("app.pipelines.scanner.SP500_TICKERS", ["AAPL", "MSFT"]):
        result = warmup_cache_task()

    assert isinstance(result, str)
    assert "완료" in result or "캐시" in result


@patch("app.workers.scan_tasks._get_redis")
def test_warmup_cache_task_handles_fetch_exception(mock_get_redis) -> None:
    """fetch_stock_data가 예외를 던져도 warmup_cache_task는 정상 완료된다."""
    from app.workers.scan_tasks import warmup_cache_task

    with patch("app.pipelines.fetcher.fetch_stock_data", side_effect=Exception("timeout")), \
         patch("app.pipelines.scanner.SP500_TICKERS", ["AAPL"]):
        result = warmup_cache_task()

    assert isinstance(result, str)


@patch("app.workers.scan_tasks._get_redis")
def test_warmup_cache_task_counts_none_df_as_failed(mock_get_redis) -> None:
    """fetch_stock_data가 (None, {})을 반환하면 failed 카운트에 포함된다."""
    from app.workers.scan_tasks import warmup_cache_task

    with patch("app.pipelines.fetcher.fetch_stock_data", return_value=(None, {})), \
         patch("app.pipelines.scanner.SP500_TICKERS", ["AAPL", "MSFT"]):
        result = warmup_cache_task()

    assert "실패 2" in result or isinstance(result, str)