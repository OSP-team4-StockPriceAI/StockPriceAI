"""
S&P 500 배치 스캐너 — Redis 캐시 기반
기존 scan_cache.json → Redis (TTL 24h)
ThreadPoolExecutor + 전체 앙상블
"""

import json
import logging
import os
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, cast

import numpy as np
import pandas as pd
import redis
import yfinance as yf

from ..core.config import PARALLEL, settings

warnings.filterwarnings("ignore")

# OMP 스레드 수 제한 (컨테이너 환경)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

log = logging.getLogger("stockai.scanner")

CACHE_TTL_H = settings.scan_cache_ttl_hours
EWMA_ALPHA = 0.3
WORKER_TIMEOUT_SEC = 600
CACHE_KEY_PREFIX = "scan:ticker:"
CACHE_KEY_PATTERN = "scan:ticker:*"


# ─────────────────────────────────────────────────────────────
# S&P 500 종목 리스트
# ─────────────────────────────────────────────────────────────

SP500_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AVGO", "META", "GOOGL", "GOOG", "TSLA", "ORCL", "CRM",
    "AMD", "QCOM", "TXN", "INTC", "ADI", "MU", "AMAT", "LRCX", "KLAC", "MRVL",
    "CDNS", "SNPS", "FTNT", "PANW", "CRWD", "NOW", "ADSK", "ANSS", "GDDY", "PAYC",
    "TTWO", "EA", "AKAM", "CTSH", "EPAM", "FFIV", "JNPR", "NTAP", "STX", "WDC",
    "HPE", "HPQ", "DELL", "CSCO", "IBM", "ACN", "INTU", "FSLR", "GLW", "KEYS",
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "BLK",
    "SCHW", "USB", "PNC", "TFC", "COF", "SPGI", "MCO", "ICE", "CME", "CBOE",
    "AON", "MMC", "AJG", "BRO", "WTW", "AFL", "MET", "PRU", "ALL", "CB",
    "LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY", "AMGN",
    "GILD", "CVS", "CI", "HUM", "ELV", "CNC", "MOH", "MDT", "SYK", "BSX",
    "EW", "ISRG", "RMD", "DXCM", "IDXX", "IQV", "BDX", "ZBH", "HOLX", "ALGN",
    "AMZN", "HD", "MCD", "NKE", "SBUX", "TJX", "LOW", "BKNG", "CMG", "MAR",
    "HLT", "CCL", "RCL", "NCLH", "MGM", "CZR", "WYNN", "LVS", "F", "GM",
    "TSCO", "ROST", "DLTR", "DG", "BBY", "KMX", "AN", "PAG", "AZO", "ORLY",
    "ULTA", "LEN", "PHM", "DHI", "NVR", "TOL",
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "KHC",
    "GIS", "K", "CPB", "SJM", "CAG", "HRL", "MKC", "CHD", "CLX", "EL",
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "DVN", "HES",
    "HAL", "BKR", "OXY", "APA", "MRO",
    "CAT", "HON", "UPS", "BA", "RTX", "LMT", "NOC", "GD", "DE", "EMR",
    "ETN", "ROK", "AME", "FTV", "CARR", "OTIS", "TDG", "HWM", "GE",
    "WM", "RSG", "WCN", "CTAS", "PAYX", "ADP", "VRSK",
    "LIN", "APD", "SHW", "FCX", "NEM", "NUE", "STLD",
    "AMT", "PLD", "EQIX", "CCI", "SPG", "PSA", "EXR", "WELL", "VTR",
    "NEE", "DUK", "SO", "D", "EXC", "AEP", "SRE", "XEL", "WEC",
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR",
]
SP500_TICKERS = list(dict.fromkeys(SP500_TICKERS))


# ─────────────────────────────────────────────────────────────
# Redis 캐시
# ─────────────────────────────────────────────────────────────

def _get_redis() -> redis.Redis:
    return cast(redis.Redis, redis.from_url(settings.redis_url, decode_responses=True))  # type: ignore[no-untyped-call]


def load_cache(ticker: str) -> dict[str, Any] | None:
    try:
        r = _get_redis()
        raw = cast("str | None", r.get(f"{CACHE_KEY_PREFIX}{ticker}"))
        if raw:
            return cast(dict[str, Any], json.loads(raw))
    except Exception:
        pass
    return None


def save_cache(ticker: str, data: dict[str, Any], ttl_hours: int = CACHE_TTL_H) -> None:
    try:
        r = _get_redis()
        r.setex(
            f"{CACHE_KEY_PREFIX}{ticker}",
            int(ttl_hours * 3600),
            json.dumps(data, default=str),
        )
    except Exception:
        pass


def load_all_cache(tickers: list[str]) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    try:
        r = _get_redis()
        pipe = r.pipeline()
        for t in tickers:
            pipe.get(f"{CACHE_KEY_PREFIX}{t}")
        results = pipe.execute()
        for t, raw in zip(tickers, results):
            if raw:
                try:
                    cache[t] = cast(dict[str, Any], json.loads(raw))
                except Exception:
                    pass
    except Exception:
        pass
    return cache


def save_all_cache(updates: dict[str, dict[str, Any]], ttl_hours: int = CACHE_TTL_H) -> None:
    try:
        r = _get_redis()
        pipe = r.pipeline()
        ttl_sec = int(ttl_hours * 3600)
        for ticker, data in updates.items():
            pipe.setex(
                f"{CACHE_KEY_PREFIX}{ticker}",
                ttl_sec,
                json.dumps(data, default=str),
            )
        pipe.execute()
    except Exception:
        pass


def is_cache_valid(entry: dict[str, Any], ttl_hours: float = CACHE_TTL_H) -> bool:
    try:
        t = datetime.fromisoformat(entry["cached_at"])
        return datetime.now() - t < timedelta(hours=ttl_hours)
    except Exception:
        return False


def get_latest_close(ticker: str) -> float | None:
    try:
        hist = yf.Ticker(ticker).history(period="2d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def price_changed(entry: dict[str, Any], threshold_pct: float = 2.0) -> bool:
    cached_price = entry.get("current_price")
    if cached_price is None:
        return True
    latest = get_latest_close(entry["ticker"])
    if latest is None:
        return False
    return bool(abs(latest - float(cached_price)) / float(cached_price) * 100 >= threshold_pct)


# ─────────────────────────────────────────────────────────────
# DP 블렌딩
# ─────────────────────────────────────────────────────────────

BLENDABLE = [
    "up_probability", "estimated_upside", "composite_score",
    "rsi", "bb_position", "momentum_pct", "atr_pct",
    "buy_signals", "sell_signals",
]


def dp_blend(old: dict[str, Any], new: dict[str, Any], alpha: float = EWMA_ALPHA) -> dict[str, Any]:
    out = new.copy()
    for field in BLENDABLE:
        if field in old and field in new:
            try:
                out[field] = round(alpha * float(new[field]) + (1 - alpha) * float(old[field]), 4)
            except (TypeError, ValueError):
                pass
    out["dp_blend_count"] = old.get("dp_blend_count", 1) + 1
    out["prev_composite_score"] = old.get("composite_score")
    return out


# ─────────────────────────────────────────────────────────────
# 단일 종목 앙상블 분석
# ─────────────────────────────────────────────────────────────

def analyze_single_ticker(ticker: str, period_days: int = 400) -> dict[str, Any] | None:
    """XGBoost + LSTM 전체 앙상블 분석."""
    t0 = time.time()
    try:
        from ..models.predictor import EnsemblePredictor
        from .fetcher import fetch_stock_data
        from .technical import add_all_indicators, get_current_signals

        df, info = fetch_stock_data(ticker, period_days=period_days)
        if df is None or info is None or len(df) < 60:
            return None

        df = add_all_indicators(df)

        pred_m = EnsemblePredictor(scanner_mode=True)
        metrics = pred_m.train(df, include_sentiment=False, force_lstm=False)
        if "error" in metrics:
            return None

        pred = pred_m.predict(df)
        if "error" in pred:
            return None

        latest = df.iloc[-1]
        close = float(df["Close"].iloc[-1])
        high_52w = float(df["High"].tail(252).max())
        upside52 = (high_52w - close) / close * 100
        atr_pct = float(latest.get("ATR_Pct", 1.0))
        exp3m = atr_pct * np.sqrt(63)
        rsi = float(latest.get("RSI14", 50))
        bb_pos = float(latest.get("BB_Position", 0.5))
        up_prob = float(pred["up_probability"])

        est_up = (
            up_prob * exp3m * 0.4
            + max(0.0, (70 - rsi) / 70) * exp3m * 0.3
            + min(upside52, 30) * 0.3
        )

        momentum = float(latest.get("Momentum_Normalized", 0))
        mom_f = (
            0.7 if momentum > 0.15 else
            0.9 if momentum > 0.05 else
            1.1 if momentum < -0.10 else 1.0
        )
        composite = up_prob * est_up * mom_f

        per = info.get("trailingPE")
        beta = float(info.get("beta", 1.0) or 1.0)
        mktcap = float(info.get("marketCap", 0) or 0)
        qf = (
            (0.8 if mktcap < 1e9 else 1.0)
            * (0.7 if per and per < 0 else 1.0)
            * (0.8 if beta > 3 else 1.0)
        )
        composite *= qf

        signals = get_current_signals(df)
        buy_signals = sum(1 for s, _, _ in signals.values() if s == "BUY")
        sell_signals = sum(1 for s, _, _ in signals.values() if s == "SELL")
        now_iso = datetime.now().isoformat()
        elapsed = time.time() - t0

        log.debug(f"✅ {ticker} 완료 ({elapsed:.1f}s)")

        return {
            "ticker": ticker,
            "period_days": period_days,
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "current_price": round(close, 2),
            "up_probability": round(up_prob * 100, 1),
            "estimated_upside": round(est_up, 1),
            "composite_score": round(composite, 4),
            "rsi": round(rsi, 1),
            "bb_position": round(bb_pos, 2),
            "momentum_pct": round(momentum * 100, 1),
            "atr_pct": round(atr_pct, 2),
            "beta": round(beta, 2),
            "market_cap": mktcap,
            "per": round(per, 1) if per else None,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "ml_signal": pred["signal"],
            "model_type": metrics.get("model_type", "N/A"),
            "upside_to_52w": round(upside52, 1),
            "analyzed_at": now_iso,
            "cached_at": now_iso,
            "dp_blend_count": 1,
            "prev_composite_score": None,
        }

    except Exception:
        log.warning(f"❌ {ticker} 분석 실패:\n{traceback.format_exc()}")
        return None


# ─────────────────────────────────────────────────────────────
# 캐시 처리 래퍼
# ─────────────────────────────────────────────────────────────

def _process_one(
    ticker: str,
    old: dict[str, Any] | None,
    force_refresh: bool,
    price_threshold_pct: float,
    period_days: int = 400,
    progress_callback: Any = None,
) -> tuple[str, dict[str, Any] | None, str]:
    if old and old.get("period_days", 400) != period_days:
        force_refresh = True

    if old and not force_refresh and is_cache_valid(old, CACHE_TTL_H):
        if not price_changed(old, price_threshold_pct):
            return ticker, old, "cached"

    new = analyze_single_ticker(ticker, period_days=period_days)
    if new is None:
        return ticker, old, "failed"

    blended = dp_blend(old, new) if old else new
    return ticker, blended, "refreshed"


# ─────────────────────────────────────────────────────────────
# 진행 상태 객체
# ─────────────────────────────────────────────────────────────

class ScanProgress:
    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.cached = 0
        self.refreshed = 0
        self.failed = 0
        self.current_ticker = ""
        self.live_results: list[dict[str, Any]] = []
        self.started_at = datetime.now()

    @property
    def pct(self) -> float:
        return self.done / self.total if self.total else 0.0

    @property
    def elapsed_sec(self) -> float:
        return (datetime.now() - self.started_at).total_seconds()

    @property
    def eta_sec(self) -> float | None:
        if self.done == 0:
            return None
        rate = self.done / max(self.elapsed_sec, 0.001)
        return (self.total - self.done) / rate if rate > 0 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "done": self.done,
            "pct": round(self.pct * 100, 1),
            "cached": self.cached,
            "refreshed": self.refreshed,
            "failed": self.failed,
            "current_ticker": self.current_ticker,
            "elapsed_sec": round(self.elapsed_sec, 1),
            "eta_sec": round(self.eta_sec, 1) if self.eta_sec else None,
        }

    def sorted_df(self) -> pd.DataFrame:
        if not self.live_results:
            return pd.DataFrame()
        df = pd.DataFrame(self.live_results)
        return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 메인 배치 스캐너
# ─────────────────────────────────────────────────────────────

def run_sp500_scan(
    tickers: list[str],
    max_workers: int = 0,
    force_refresh: bool = False,
    price_threshold_pct: float = 2.0,
    stop_flag: dict[str, Any] | None = None,
    progress: ScanProgress | None = None,
    period_days: int = 400,
    progress_callback: Any = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """ThreadPoolExecutor 기반 배치 스캔 (Redis 캐시 사용)."""
    if max_workers <= 0:
        max_workers = PARALLEL["scanner_workers"]

    # 배치로 캐시 조회
    cache = load_all_cache(tickers)
    t_scan = time.time()

    need_analysis = sum(
        1 for t in tickers
        if force_refresh
        or t not in cache
        or not is_cache_valid(cache.get(t, {}), CACHE_TTL_H)
    )
    log.info(
        f"스캔 시작: 종목={len(tickers)}, 워커={max_workers}, "
        f"캐시재사용={len(tickers)-need_analysis}, 신규분석={need_analysis}"
    )

    pending_saves: dict[str, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_one,
                t,
                cache.get(t),
                force_refresh,
                price_threshold_pct,
                period_days,
                progress_callback,
            ): t
            for t in tickers
        }

        for future in as_completed(futures):
            if stop_flag and stop_flag.get("stop"):
                log.warning("스캔 중단 요청")
                executor.shutdown(wait=False, cancel_futures=True)
                break

            ticker = futures[future]
            if progress:
                progress.current_ticker = ticker

            try:
                _, result, status = future.result(timeout=WORKER_TIMEOUT_SEC)
            except TimeoutError:
                result, status = None, "failed"
                log.warning(f"{ticker} 타임아웃 ({WORKER_TIMEOUT_SEC}s)")
            except Exception as e:
                result, status = None, "failed"
                log.warning(f"{ticker} 오류: {e}")

            if progress:
                progress.done += 1
                if status == "cached":
                    progress.cached += 1
                elif status == "refreshed":
                    progress.refreshed += 1
                else:
                    progress.failed += 1

            if result is not None:
                cache[ticker] = result
                pending_saves[ticker] = result
                if progress:
                    progress.live_results.append(result)

                if progress_callback:
                    try:
                        progress_callback(
                            progress.to_dict() if progress else {"ticker": ticker, "status": status}
                        )
                    except Exception:
                        pass

            # 10개마다 Redis에 일괄 저장
            if len(pending_saves) >= 10:
                save_all_cache(pending_saves)
                pending_saves.clear()

    if pending_saves:
        save_all_cache(pending_saves)

    elapsed = time.time() - t_scan
    n_ok = sum(1 for t in tickers if t in cache)
    log.info(f"스캔 완료: {elapsed/60:.1f}분, 성공={n_ok}종목")

    all_res = [cache[t] for t in tickers if t in cache]
    if not all_res:
        return pd.DataFrame(), cache

    df = pd.DataFrame(all_res)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.index += 1
    return df, cache


def get_cache_stats(tickers: list[str]) -> dict[str, Any]:
    cache = load_all_cache(tickers)
    valid = stale = 0
    for t in tickers:
        e = cache.get(t)
        if e:
            if is_cache_valid(e):
                valid += 1
            else:
                stale += 1
    return {
        "total_cached": len([t for t in tickers if t in cache]),
        "valid": valid,
        "stale": stale,
        "uncached": len([t for t in tickers if t not in cache]),
        "ttl_hours": CACHE_TTL_H,
    }
