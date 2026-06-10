"""
Celery 비동기 스캔 태스크
S&P 500 배치 스캔을 백그라운드에서 실행하고
Redis에 진행률 및 결과를 저장합니다.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

import redis

from ..core.config import settings
from .celery_app import celery_app

log = logging.getLogger("stockai.tasks")

PROGRESS_KEY_PREFIX = "scan:progress:"
PROGRESS_TTL = 86400  # 24h

# 자동 스캔 결과의 최신 job_id 를 저장하는 Redis 키
LATEST_SCAN_KEY = "scan:latest:{trigger}"  # trigger = market_open | market_close

ET = ZoneInfo("America/New_York")


def _get_redis() -> redis.Redis:
    return redis.from_url(settings.redis_url, decode_responses=True)  # type: ignore[no-untyped-call, no-any-return]


def _save_progress(job_id: str, data: dict[str, Any]) -> None:
    try:
        r = _get_redis()
        r.setex(f"{PROGRESS_KEY_PREFIX}{job_id}", PROGRESS_TTL, json.dumps(data, default=str))
    except Exception:
        pass


def get_scan_progress(job_id: str) -> dict[str, Any] | None:
    try:
        r = _get_redis()
        raw = cast("str | None", r.get(f"{PROGRESS_KEY_PREFIX}{job_id}"))
        if raw:
            return cast(dict[str, Any], json.loads(raw))
    except Exception:
        pass
    return None


def get_latest_scan_job_id(trigger: str) -> str | None:
    """market_open / market_close 트리거의 최신 job_id 조회"""
    try:
        r = _get_redis()
        return cast("str | None", r.get(LATEST_SCAN_KEY.format(trigger=trigger)))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# 핵심 스캔 태스크
# ─────────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="scan_tasks.run_scan_job", max_retries=0)  # type: ignore[untyped-decorator]
def run_scan_job(
    self: Any,
    job_id: str,
    tickers: list[str],
    max_workers: int = 2,
    force_refresh: bool = False,
    period_days: int = 400,
) -> dict[str, Any]:
    """
    S&P 500 배치 스캔 Celery 태스크.

    진행률을 Redis에 저장하고, 완료 시 결과도 Redis에 저장합니다.
    WebSocket 엔드포인트는 이 Redis 키를 폴링하여 클라이언트에 전달합니다.
    """
    log.info(f"스캔 시작: job_id={job_id}, 종목={len(tickers)}")

    # 초기 상태 저장
    _save_progress(
        job_id,
        {
            "job_id": job_id,
            "status": "running",
            "total": len(tickers),
            "done": 0,
            "pct": 0.0,
            "cached": 0,
            "refreshed": 0,
            "failed": 0,
            "current_ticker": "",
            "started_at": datetime.now().isoformat(),
            "results": [],
        },
    )

    try:
        from ..pipelines.scanner import ScanProgress, run_sp500_scan

        progress = ScanProgress(total=len(tickers))

        def on_progress(state: dict[str, Any]) -> None:
            _save_progress(
                job_id,
                {
                    "job_id": job_id,
                    "status": "running",
                    **state,
                    "results": [
                        {
                            k: v
                            for k, v in r.items()
                            if k
                            in (
                                "ticker",
                                "name",
                                "sector",
                                "composite_score",
                                "up_probability",
                                "estimated_upside",
                                "ml_signal",
                                "current_price",
                                "rsi",
                            )
                        }
                        for r in progress.live_results[-20:]  # 최근 20개만
                    ],
                },
            )

        scan_df, _ = run_sp500_scan(
            tickers=tickers,
            max_workers=max_workers,
            force_refresh=force_refresh,
            period_days=period_days,
            progress=progress,
            progress_callback=on_progress,
        )

        # 최종 결과 요약
        top_results = []
        if not scan_df.empty:
            top_results = (
                scan_df.head(50)[
                    [
                        "ticker",
                        "name",
                        "sector",
                        "composite_score",
                        "up_probability",
                        "estimated_upside",
                        "ml_signal",
                        "current_price",
                        "rsi",
                        "buy_signals",
                        "market_cap",
                    ]
                ]
                .fillna(0)
                .to_dict("records")
            )

        final_state = {
            "job_id": job_id,
            "status": "completed",
            "total": len(tickers),
            "done": progress.done,
            "pct": 100.0,
            "cached": progress.cached,
            "refreshed": progress.refreshed,
            "failed": progress.failed,
            "current_ticker": "",
            "elapsed_sec": round(progress.elapsed_sec, 1),
            "eta_sec": None,
            "results": top_results,
            "completed_at": datetime.now().isoformat(),
        }
        _save_progress(job_id, final_state)
        log.info(f"스캔 완료: job_id={job_id}, 성공={progress.refreshed + progress.cached}")
        return final_state

    except Exception as e:
        error_state = {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat(),
        }
        _save_progress(job_id, error_state)
        log.error(f"스캔 실패: job_id={job_id}, error={e}")
        raise


# ─────────────────────────────────────────────────────────────
# 스케줄 트리거 태스크 (Celery Beat → 이 태스크 → run_scan_job)
# ─────────────────────────────────────────────────────────────

@celery_app.task(name="scan_tasks.scheduled_market_scan", ignore_result=True)  # type: ignore[untyped-decorator]
def scheduled_market_scan(
    trigger: str = "market_open",
    force_refresh: bool = True,
    max_workers: int = 2,
    period_days: int = 400,
) -> None:
    """
    Celery Beat이 장 시작 +30분 / 장 마감 -30분에 자동 호출하는 태스크.

    trigger: "market_open" | "market_close"

    흐름:
      1. 미국 공휴일/주말 체크 → 장 휴장이면 즉시 종료
      2. 새 job_id 생성 후 run_scan_job 비동기 실행
      3. latest job_id 를 Redis에 기록 (API에서 조회 가능)
    """
    now_et = datetime.now(ET)
    log.info(f"[{trigger}] 자동 스캔 트리거 — {now_et.strftime('%Y-%m-%d %H:%M %Z')}")

    # ── 주말 체크 ────────────────────────────────────────────
    if now_et.weekday() >= 5:  # 5=토, 6=일
        log.info(f"[{trigger}] 주말 휴장 — 스캔 건너뜀")
        return

    # ── 미국 주요 공휴일 체크 ────────────────────────────────
    if _is_us_market_holiday(now_et):
        log.info(f"[{trigger}] 공휴일 휴장 — 스캔 건너뜀")
        return

    # ── S&P 500 종목 목록 로드 ───────────────────────────────
    try:
        from ..pipelines.scanner import SP500_TICKERS
        tickers = SP500_TICKERS
    except Exception as e:
        log.error(f"[{trigger}] 종목 목록 로드 실패: {e}")
        return

    if not tickers:
        log.error(f"[{trigger}] 종목 목록이 비어 있음")
        return

    # ── job 생성 및 디스패치 ─────────────────────────────────
    job_id = str(uuid.uuid4())

    _save_progress(
        job_id,
        {
            "job_id": job_id,
            "status": "queued",
            "trigger": trigger,
            "scheduled_at": now_et.isoformat(),
            "total": len(tickers),
            "done": 0,
            "pct": 0.0,
            "cached": 0,
            "refreshed": 0,
            "failed": 0,
            "current_ticker": "",
        },
    )

    # 최신 job_id 를 Redis에 기록 (TTL 48h)
    try:
        r = _get_redis()
        r.setex(LATEST_SCAN_KEY.format(trigger=trigger), 172800, job_id)
    except Exception as e:
        log.warning(f"[{trigger}] latest key 저장 실패: {e}")

    run_scan_job.apply_async(
        kwargs={
            "job_id": job_id,
            "tickers": tickers,
            "max_workers": max_workers,
            "force_refresh": force_refresh,
            "period_days": period_days,
        },
        task_id=job_id,
    )

    log.info(
        f"[{trigger}] 자동 스캔 시작 — job_id={job_id}, 종목={len(tickers)}개, "
        f"ET={now_et.strftime('%H:%M')}"
    )


# ─────────────────────────────────────────────────────────────
# 미국 공휴일 유틸
# ─────────────────────────────────────────────────────────────

def _is_us_market_holiday(dt: datetime) -> bool:
    """
    NYSE 주요 공휴일 여부를 반환합니다.
    고정 공휴일 + 부활절(변동) 커버.
    완전한 정확도가 필요하다면 `pandas_market_calendars` 사용을 권장합니다.
    """
    year, month, day = dt.year, dt.month, dt.day

    # 고정 공휴일 (월요일 대체 포함하지 않음 — 아래 _observed 로직에서 처리)
    fixed_holidays = {
        (1, 1),    # 새해
        (7, 4),    # 독립기념일
        (12, 25),  # 크리스마스
    }

    # 변동 공휴일 계산
    variable_holidays = {
        _mlk_day(year),          # 1월 셋째 월요일
        _presidents_day(year),   # 2월 셋째 월요일
        _good_friday(year),      # 부활절 전 금요일
        _memorial_day(year),     # 5월 마지막 월요일
        _juneteenth(year),       # 6월 19일 (2022년~)
        _labor_day(year),        # 9월 첫째 월요일
        _thanksgiving(year),     # 11월 넷째 목요일
    }

    if (month, day) in fixed_holidays:
        return True

    # 토요일 공휴일 → 금요일 대체 / 일요일 → 월요일 대체
    for h_month, h_day in fixed_holidays:
        h = datetime(year, h_month, h_day)
        if h.weekday() == 5 and (month, day) == (h_month, h_day - 1):  # 토 → 금
            return True
        if h.weekday() == 6 and (month, day) == (h_month, h_day + 1):  # 일 → 월
            return True

    return (month, day) in variable_holidays


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> tuple[int, int]:
    """해당 월의 n번째 weekday (0=월 … 6=일) → (month, day)"""
    count = 0
    for d in range(1, 32):
        try:
            dt = datetime(year, month, d)
        except ValueError:
            break
        if dt.weekday() == weekday:
            count += 1
            if count == n:
                return (month, d)
    return (month, 1)  # fallback (발생 안 함)


def _last_weekday(year: int, month: int, weekday: int) -> tuple[int, int]:
    """해당 월의 마지막 weekday → (month, day)"""
    result = (month, 1)
    for d in range(1, 32):
        try:
            dt = datetime(year, month, d)
        except ValueError:
            break
        if dt.weekday() == weekday:
            result = (month, d)
    return result


def _mlk_day(year: int) -> tuple[int, int]:
    return _nth_weekday(year, 1, 0, 3)   # 1월 셋째 월요일


def _presidents_day(year: int) -> tuple[int, int]:
    return _nth_weekday(year, 2, 0, 3)   # 2월 셋째 월요일


def _memorial_day(year: int) -> tuple[int, int]:
    return _last_weekday(year, 5, 0)      # 5월 마지막 월요일


def _labor_day(year: int) -> tuple[int, int]:
    return _nth_weekday(year, 9, 0, 1)   # 9월 첫째 월요일


def _thanksgiving(year: int) -> tuple[int, int]:
    return _nth_weekday(year, 11, 3, 4)  # 11월 넷째 목요일


def _juneteenth(year: int) -> tuple[int, int]:
    if year < 2022:
        return (6, 99)  # 존재 안 함 (매칭 불가 값)
    return (6, 19)


def _good_friday(year: int) -> tuple[int, int]:
    """부활절 전 금요일 (Gauss 알고리즘)"""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    # 부활절 → 2일 전 = 금요일
    easter = datetime(year, month, day)
    from datetime import timedelta
    gf = easter - timedelta(days=2)
    return (gf.month, gf.day)


# ─────────────────────────────────────────────────────────────
# 기존: 캐시 웜업 태스크
# ─────────────────────────────────────────────────────────────

@celery_app.task(name="scan_tasks.warmup_cache_task", ignore_result=True)  # type: ignore[untyped-decorator]
def warmup_cache_task() -> str:
    """
    주기적으로 S&P 500 종목의 데이터를 수집하여 Redis 캐시를 최신화(웜업)합니다.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from ..pipelines.fetcher import fetch_stock_data
    from ..pipelines.scanner import SP500_TICKERS

    log.info(f"🔄 캐시 웜업 시작: {len(SP500_TICKERS)} 종목")

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(fetch_stock_data, ticker, 400, True): ticker
            for ticker in SP500_TICKERS
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                df, _ = future.result(timeout=30)
                if df is not None:
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                log.warning(f"⚠️ 웜업 실패 [{ticker}]: {e}")

    result_msg = f"✅ 캐시 웜업 완료: 성공 {success}, 실패 {failed}"
    log.info(result_msg)
    return result_msg