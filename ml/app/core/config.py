"""
ML 서비스 설정 — 환경 변수 기반 범용 설정
Apple Silicon 특화 코드 제거, CPU 모드로 EC2 컨테이너 환경 지원
"""

from __future__ import annotations

import multiprocessing
import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Redis
    redis_url: str = "redis://redis:6379/1"
    celery_broker_url: str = "redis://redis:6379/2"

    # Celery
    celery_result_expires: int = 86400  # 24h

    # Scanner
    scanner_workers: int = int(os.getenv("SCANNER_WORKERS", "2"))
    scan_cache_ttl_hours: int = 24

    # ML
    max_train_samples: int = int(os.getenv("MAX_TRAIN_SAMPLES", "4000"))
    max_lstm_samples: int = int(os.getenv("MAX_LSTM_SAMPLES", "3000"))
    lstm_device: str = os.getenv("LSTM_DEVICE", "cpu")  # EC2에는 GPU 없음

    # Backend 서비스 URL (감정지수 저장/조회)
    backend_url: str = os.getenv("BACKEND_URL", "http://backend:8000")

    # CORS
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://backend:8000",
    ]

    model_config = {"env_file": ".env", "case_sensitive": False}


settings = Settings()

# ─────────────────────────────────────────────────────────────
# 플랫폼 독립 XGBoost 설정
# ─────────────────────────────────────────────────────────────

_cpu_count = multiprocessing.cpu_count()

XGBOOST = {
    "tree_method": "hist",
    "device": "cpu",
    "nthread": _cpu_count,
    "max_bin": 256,
    "grow_policy": "lossguide",
}

# 스캐너 모드: OMP 경합 방지용 1스레드 고정
XGBOOST_SCANNER = {
    "tree_method": "hist",
    "device": "cpu",
    "nthread": 1,
    "max_bin": 128,
    "grow_policy": "lossguide",
}

# PyTorch 스캐너 전용 — CPU 강제 (MPS 없음)
PYTORCH_SCANNER = {
    "device": "cpu",
    "dtype": "float32",
    "num_threads": 1,
    "batch_size": 32,
}

# PyTorch 일반 — 환경변수로 장치 결정
PYTORCH = {
    "device": settings.lstm_device,
    "dtype": "float32",
    "num_threads": _cpu_count,
    "batch_size": 64,
}

DATA = {
    "max_train_samples": settings.max_train_samples,
    "max_lstm_samples": settings.max_lstm_samples,
}

PARALLEL = {
    "scanner_workers": settings.scanner_workers,
}


def get_torch_device() -> str:
    """최적 PyTorch 디바이스 반환 (CUDA → CPU 순)."""
    if settings.lstm_device != "cpu":
        try:
            import torch

            if settings.lstm_device == "cuda" and torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    return "cpu"