import logging

from celery import Celery
from celery.schedules import crontab

from ..core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

celery_app = Celery(
    "stockai_ml",
    broker=settings.celery_broker_url,
    backend=settings.redis_url,
    include=["app.workers.scan_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # 미국 동부시간(ET) 기준 — NYSE 장 시간대
    timezone="America/New_York",
    enable_utc=True,
    result_expires=settings.celery_result_expires,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    beat_schedule={
        # ── 장 시작 후 30분 (09:30 ET 개장 → 10:00 ET 실행) ──────────
        # 월~금, 미국 동부시간 10:00
        "market-open-scan": {
            "task": "scan_tasks.scheduled_market_scan",
            "schedule": crontab(hour=10, minute=0, day_of_week="1-5"),
            "kwargs": {
                "trigger": "market_open",
                "force_refresh": True,
            },
        },
        # ── 장 마감 30분 전 (16:00 ET 마감 → 15:30 ET 실행) ──────────
        # 월~금, 미국 동부시간 15:30
        "market-close-scan": {
            "task": "scan_tasks.scheduled_market_scan",
            "schedule": crontab(hour=15, minute=30, day_of_week="1-5"),
            "kwargs": {
                "trigger": "market_close",
                "force_refresh": True,
            },
        },
        # ── 기존: 캐시 웜업 (매 1시간) ───────────────────────────────
        "warmup-sp500-cache-every-hour": {
            "task": "scan_tasks.warmup_cache_task",
            "schedule": 3600.0,
        },
    },
)