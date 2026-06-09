"""
scanner.py 단위 테스트

외부 의존성(Redis, yfinance, ML 모델)은 전부 mock 처리.
테스트 대상:
  - SP500_TICKERS
  - is_cache_valid
  - dp_blend
  - load_cache / save_cache
  - load_all_cache / save_all_cache
  - get_latest_close / price_changed
  - ScanProgress (pct, elapsed_sec, eta_sec, to_dict, sorted_df)
  - _process_one
  - get_cache_stats
  - analyze_single_ticker
  - run_sp500_scan
"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────
# 픽스처 / 헬퍼
# ─────────────────────────────────────────────────────────────

def _entry(
    ticker: str = "AAPL",
    cached_at: datetime | None = None,
    hours_old: float = 0.0,
    composite_score: float = 1.0,
    current_price: float = 150.0,
    period_days: int = 400,
) -> dict:
    """테스트용 캐시 엔트리."""
    ts = (cached_at or datetime.now()) - timedelta(hours=hours_old)
    return {
        "ticker": ticker,
        "cached_at": ts.isoformat(),
        "composite_score": composite_score,
        "current_price": current_price,
        "period_days": period_days,
        "up_probability": 60.0,
        "estimated_upside": 5.0,
        "rsi": 50.0,
        "bb_position": 0.5,
        "momentum_pct": 1.0,
        "atr_pct": 1.5,
        "buy_signals": 3,
        "sell_signals": 1,
        "dp_blend_count": 1,
        "prev_composite_score": None,
    }


def _mock_redis(get_val=None, pipeline_vals=None) -> MagicMock:
    """Redis 클라이언트 mock."""
    r = MagicMock()
    r.get.return_value = get_val
    pipe = MagicMock()
    pipe.__enter__ = lambda s: s
    pipe.__exit__ = MagicMock(return_value=False)
    pipe.execute.return_value = pipeline_vals or []
    r.pipeline.return_value = pipe
    return r


# ─────────────────────────────────────────────────────────────
# SP500_TICKERS
# ─────────────────────────────────────────────────────────────

class TestSP500Tickers:
    def test_is_non_empty_list(self):
        from app.pipelines.scanner import SP500_TICKERS
        assert isinstance(SP500_TICKERS, list)
        assert len(SP500_TICKERS) > 0

    def test_all_elements_are_strings(self):
        from app.pipelines.scanner import SP500_TICKERS
        assert all(isinstance(t, str) for t in SP500_TICKERS)

    def test_no_duplicates(self):
        from app.pipelines.scanner import SP500_TICKERS
        assert len(SP500_TICKERS) == len(set(SP500_TICKERS))

    def test_contains_major_tickers(self):
        from app.pipelines.scanner import SP500_TICKERS
        for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]:
            assert ticker in SP500_TICKERS

    def test_no_empty_strings(self):
        from app.pipelines.scanner import SP500_TICKERS
        assert all(t.strip() for t in SP500_TICKERS)


# ─────────────────────────────────────────────────────────────
# is_cache_valid
# ─────────────────────────────────────────────────────────────

class TestIsCacheValid:
    def test_fresh_entry_is_valid(self):
        from app.pipelines.scanner import is_cache_valid
        entry = _entry(hours_old=1.0)
        assert is_cache_valid(entry, ttl_hours=24) is True

    def test_expired_entry_is_invalid(self):
        from app.pipelines.scanner import is_cache_valid
        entry = _entry(hours_old=25.0)
        assert is_cache_valid(entry, ttl_hours=24) is False

    def test_exactly_at_boundary_is_invalid(self):
        from app.pipelines.scanner import is_cache_valid
        entry = _entry(hours_old=24.0)
        assert is_cache_valid(entry, ttl_hours=24) is False

    def test_just_under_boundary_is_valid(self):
        from app.pipelines.scanner import is_cache_valid
        entry = _entry(hours_old=23.9)
        assert is_cache_valid(entry, ttl_hours=24) is True

    def test_missing_cached_at_returns_false(self):
        from app.pipelines.scanner import is_cache_valid
        assert is_cache_valid({}, ttl_hours=24) is False

    def test_malformed_cached_at_returns_false(self):
        from app.pipelines.scanner import is_cache_valid
        assert is_cache_valid({"cached_at": "not-a-date"}, ttl_hours=24) is False

    def test_custom_ttl_respected(self):
        from app.pipelines.scanner import is_cache_valid
        entry = _entry(hours_old=2.0)
        assert is_cache_valid(entry, ttl_hours=1) is False
        assert is_cache_valid(entry, ttl_hours=3) is True


# ─────────────────────────────────────────────────────────────
# dp_blend
# ─────────────────────────────────────────────────────────────

class TestDpBlend:
    def test_blendable_fields_are_ewma_weighted(self):
        from app.pipelines.scanner import dp_blend
        old = _entry(composite_score=2.0, current_price=100.0)
        new = _entry(composite_score=4.0, current_price=110.0)
        result = dp_blend(old, new, alpha=0.5)
        # alpha=0.5: 0.5*4 + 0.5*2 = 3.0
        assert result["composite_score"] == pytest.approx(3.0, abs=1e-3)

    def test_non_blendable_fields_taken_from_new(self):
        from app.pipelines.scanner import dp_blend
        old = _entry(ticker="AAPL")
        new = _entry(ticker="MSFT")
        result = dp_blend(old, new)
        assert result["ticker"] == "MSFT"

    def test_blend_count_increments(self):
        from app.pipelines.scanner import dp_blend
        old = {**_entry(), "dp_blend_count": 3}
        new = _entry()
        result = dp_blend(old, new)
        assert result["dp_blend_count"] == 4

    def test_prev_composite_score_set_from_old(self):
        from app.pipelines.scanner import dp_blend
        old = _entry(composite_score=5.0)
        new = _entry(composite_score=7.0)
        result = dp_blend(old, new)
        assert result["prev_composite_score"] == 5.0

    def test_missing_field_in_old_skips_blend(self):
        from app.pipelines.scanner import dp_blend
        old = {}
        new = _entry(composite_score=3.0)
        result = dp_blend(old, new)
        # old에 composite_score 없으면 new 값 그대로
        assert result["composite_score"] == 3.0

    def test_missing_field_in_new_skips_blend(self):
        from app.pipelines.scanner import dp_blend
        old = _entry(composite_score=2.0)
        new = {k: v for k, v in _entry(composite_score=4.0).items() if k != "composite_score"}
        result = dp_blend(old, new)
        assert "composite_score" not in result

    def test_alpha_zero_keeps_old_values(self):
        from app.pipelines.scanner import dp_blend
        old = _entry(composite_score=2.0)
        new = _entry(composite_score=8.0)
        result = dp_blend(old, new, alpha=0.0)
        assert result["composite_score"] == pytest.approx(2.0, abs=1e-3)

    def test_alpha_one_takes_new_values(self):
        from app.pipelines.scanner import dp_blend
        old = _entry(composite_score=2.0)
        new = _entry(composite_score=8.0)
        result = dp_blend(old, new, alpha=1.0)
        assert result["composite_score"] == pytest.approx(8.0, abs=1e-3)

    def test_non_numeric_field_is_skipped_gracefully(self):
        from app.pipelines.scanner import dp_blend
        old = {**_entry(), "composite_score": "invalid"}
        new = _entry(composite_score=3.0)
        result = dp_blend(old, new)
        # TypeError 발생 시 new 값 그대로
        assert result["composite_score"] == 3.0


# ─────────────────────────────────────────────────────────────
# load_cache / save_cache
# ─────────────────────────────────────────────────────────────

class TestLoadCache:
    def test_returns_dict_when_key_exists(self):
        from app.pipelines.scanner import load_cache
        data = _entry("AAPL")
        mock_r = _mock_redis(get_val=json.dumps(data))
        with patch("app.pipelines.scanner._get_redis", return_value=mock_r):
            result = load_cache("AAPL")
        assert result is not None
        assert result["ticker"] == "AAPL"

    def test_returns_none_when_key_missing(self):
        from app.pipelines.scanner import load_cache
        mock_r = _mock_redis(get_val=None)
        with patch("app.pipelines.scanner._get_redis", return_value=mock_r):
            result = load_cache("AAPL")
        assert result is None

    def test_returns_none_on_redis_exception(self):
        from app.pipelines.scanner import load_cache
        with patch("app.pipelines.scanner._get_redis", side_effect=Exception("connection error")):
            result = load_cache("AAPL")
        assert result is None

    def test_uses_correct_cache_key_prefix(self):
        from app.pipelines.scanner import CACHE_KEY_PREFIX, load_cache
        mock_r = _mock_redis(get_val=None)
        with patch("app.pipelines.scanner._get_redis", return_value=mock_r):
            load_cache("AAPL")
        mock_r.get.assert_called_once_with(f"{CACHE_KEY_PREFIX}AAPL")


class TestSaveCache:
    def test_calls_setex_with_correct_key(self):
        from app.pipelines.scanner import CACHE_KEY_PREFIX, save_cache
        mock_r = _mock_redis()
        with patch("app.pipelines.scanner._get_redis", return_value=mock_r):
            save_cache("AAPL", _entry("AAPL"), ttl_hours=24)
        key = mock_r.setex.call_args[0][0]
        assert key == f"{CACHE_KEY_PREFIX}AAPL"

    def test_calls_setex_with_correct_ttl(self):
        from app.pipelines.scanner import save_cache
        mock_r = _mock_redis()
        with patch("app.pipelines.scanner._get_redis", return_value=mock_r):
            save_cache("AAPL", _entry("AAPL"), ttl_hours=2)
        ttl_arg = mock_r.setex.call_args[0][1]
        assert ttl_arg == 2 * 3600

    def test_saved_value_is_valid_json(self):
        from app.pipelines.scanner import save_cache
        mock_r = _mock_redis()
        with patch("app.pipelines.scanner._get_redis", return_value=mock_r):
            save_cache("AAPL", _entry("AAPL"))
        json_str = mock_r.setex.call_args[0][2]
        parsed = json.loads(json_str)
        assert parsed["ticker"] == "AAPL"

    def test_silently_ignores_redis_exception(self):
        from app.pipelines.scanner import save_cache
        with patch("app.pipelines.scanner._get_redis", side_effect=Exception("error")):
            save_cache("AAPL", _entry("AAPL"))  # 예외 없이 통과


# ─────────────────────────────────────────────────────────────
# load_all_cache / save_all_cache
# ─────────────────────────────────────────────────────────────

class TestLoadAllCache:
    def test_returns_dict_with_parsed_entries(self):
        from app.pipelines.scanner import load_all_cache
        aapl = _entry("AAPL")
        msft = _entry("MSFT")
        pipeline_vals = [json.dumps(aapl), json.dumps(msft)]
        mock_r = _mock_redis(pipeline_vals=pipeline_vals)
        with patch("app.pipelines.scanner._get_redis", return_value=mock_r):
            result = load_all_cache(["AAPL", "MSFT"])
        assert "AAPL" in result and "MSFT" in result

    def test_skips_missing_tickers(self):
        from app.pipelines.scanner import load_all_cache
        pipeline_vals = [json.dumps(_entry("AAPL")), None]
        mock_r = _mock_redis(pipeline_vals=pipeline_vals)
        with patch("app.pipelines.scanner._get_redis", return_value=mock_r):
            result = load_all_cache(["AAPL", "MSFT"])
        assert "AAPL" in result
        assert "MSFT" not in result

    def test_returns_empty_dict_on_redis_exception(self):
        from app.pipelines.scanner import load_all_cache
        with patch("app.pipelines.scanner._get_redis", side_effect=Exception("error")):
            result = load_all_cache(["AAPL", "MSFT"])
        assert result == {}

    def test_skips_malformed_json(self):
        from app.pipelines.scanner import load_all_cache
        pipeline_vals = ["not-json", json.dumps(_entry("MSFT"))]
        mock_r = _mock_redis(pipeline_vals=pipeline_vals)
        with patch("app.pipelines.scanner._get_redis", return_value=mock_r):
            result = load_all_cache(["AAPL", "MSFT"])
        assert "AAPL" not in result
        assert "MSFT" in result


class TestSaveAllCache:
    def test_calls_pipeline_setex_for_each_ticker(self):
        from app.pipelines.scanner import save_all_cache
        mock_r = _mock_redis()
        updates = {"AAPL": _entry("AAPL"), "MSFT": _entry("MSFT")}
        with patch("app.pipelines.scanner._get_redis", return_value=mock_r):
            save_all_cache(updates, ttl_hours=24)
        pipe = mock_r.pipeline.return_value
        assert pipe.setex.call_count == 2
        pipe.execute.assert_called_once()

    def test_silently_ignores_redis_exception(self):
        from app.pipelines.scanner import save_all_cache
        with patch("app.pipelines.scanner._get_redis", side_effect=Exception("error")):
            save_all_cache({"AAPL": _entry("AAPL")})  # 예외 없이 통과


# ─────────────────────────────────────────────────────────────
# get_latest_close / price_changed
# ─────────────────────────────────────────────────────────────

class TestGetLatestClose:
    def test_returns_float_on_success(self):
        from app.pipelines.scanner import get_latest_close
        hist = pd.DataFrame({"Close": [145.0, 150.0]})
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist
        with patch("app.pipelines.scanner.yf.Ticker", return_value=mock_ticker):
            result = get_latest_close("AAPL")
        assert result == 150.0

    def test_returns_none_on_empty_history(self):
        from app.pipelines.scanner import get_latest_close
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        with patch("app.pipelines.scanner.yf.Ticker", return_value=mock_ticker):
            result = get_latest_close("AAPL")
        assert result is None

    def test_returns_none_on_exception(self):
        from app.pipelines.scanner import get_latest_close
        with patch("app.pipelines.scanner.yf.Ticker", side_effect=Exception("error")):
            result = get_latest_close("AAPL")
        assert result is None


class TestPriceChanged:
    def test_returns_true_when_price_moved_above_threshold(self):
        from app.pipelines.scanner import price_changed
        entry = _entry("AAPL", current_price=100.0)
        with patch("app.pipelines.scanner.get_latest_close", return_value=103.0):
            assert price_changed(entry, threshold_pct=2.0) is True

    def test_returns_false_when_price_within_threshold(self):
        from app.pipelines.scanner import price_changed
        entry = _entry("AAPL", current_price=100.0)
        with patch("app.pipelines.scanner.get_latest_close", return_value=101.0):
            assert price_changed(entry, threshold_pct=2.0) is False

    def test_returns_true_when_cached_price_missing(self):
        from app.pipelines.scanner import price_changed
        entry = {k: v for k, v in _entry("AAPL").items() if k != "current_price"}
        with patch("app.pipelines.scanner.get_latest_close", return_value=150.0):
            assert price_changed(entry) is True

    def test_returns_false_when_latest_close_unavailable(self):
        from app.pipelines.scanner import price_changed
        entry = _entry("AAPL", current_price=100.0)
        with patch("app.pipelines.scanner.get_latest_close", return_value=None):
            assert price_changed(entry) is False

    def test_exact_threshold_boundary_is_true(self):
        from app.pipelines.scanner import price_changed
        entry = _entry("AAPL", current_price=100.0)
        # 정확히 2%: abs(102-100)/100*100 == 2.0 → True (>=)
        with patch("app.pipelines.scanner.get_latest_close", return_value=102.0):
            assert price_changed(entry, threshold_pct=2.0) is True


# ─────────────────────────────────────────────────────────────
# ScanProgress
# ─────────────────────────────────────────────────────────────

class TestScanProgress:
    def test_pct_is_zero_at_start(self):
        from app.pipelines.scanner import ScanProgress
        p = ScanProgress(total=10)
        assert p.pct == 0.0

    def test_pct_updates_correctly(self):
        from app.pipelines.scanner import ScanProgress
        p = ScanProgress(total=10)
        p.done = 5
        assert p.pct == 0.5

    def test_pct_is_zero_when_total_is_zero(self):
        from app.pipelines.scanner import ScanProgress
        p = ScanProgress(total=0)
        assert p.pct == 0.0

    def test_elapsed_sec_increases_over_time(self):
        import time

        from app.pipelines.scanner import ScanProgress
        p = ScanProgress(total=10)
        time.sleep(0.05)
        assert p.elapsed_sec >= 0.04

    def test_eta_sec_is_none_before_any_progress(self):
        from app.pipelines.scanner import ScanProgress
        p = ScanProgress(total=10)
        assert p.eta_sec is None

    def test_eta_sec_is_numeric_after_progress(self):
        import time

        from app.pipelines.scanner import ScanProgress
        p = ScanProgress(total=10)
        time.sleep(0.05)
        p.done = 5
        assert p.eta_sec is not None
        assert p.eta_sec >= 0

    def test_to_dict_has_required_keys(self):
        from app.pipelines.scanner import ScanProgress
        p = ScanProgress(total=5)
        d = p.to_dict()
        required = {"total", "done", "pct", "cached", "refreshed", "failed",
                    "current_ticker", "elapsed_sec", "eta_sec"}
        assert required.issubset(d.keys())

    def test_to_dict_pct_is_percentage(self):
        from app.pipelines.scanner import ScanProgress
        p = ScanProgress(total=10)
        p.done = 4
        assert p.to_dict()["pct"] == 40.0

    def test_sorted_df_returns_empty_df_when_no_results(self):
        from app.pipelines.scanner import ScanProgress
        p = ScanProgress(total=5)
        df = p.sorted_df()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_sorted_df_sorted_by_composite_score_descending(self):
        from app.pipelines.scanner import ScanProgress
        p = ScanProgress(total=3)
        p.live_results = [
            _entry("AAPL", composite_score=1.0),
            _entry("MSFT", composite_score=3.0),
            _entry("GOOG", composite_score=2.0),
        ]
        df = p.sorted_df()
        assert list(df["ticker"]) == ["MSFT", "GOOG", "AAPL"]

    def test_counters_start_at_zero(self):
        from app.pipelines.scanner import ScanProgress
        p = ScanProgress(total=10)
        assert p.cached == 0
        assert p.refreshed == 0
        assert p.failed == 0


# ─────────────────────────────────────────────────────────────
# _process_one
# ─────────────────────────────────────────────────────────────

class TestProcessOne:
    def test_returns_cached_when_valid_and_no_price_change(self):
        from app.pipelines.scanner import _process_one
        old = _entry("AAPL", hours_old=1.0)
        with patch("app.pipelines.scanner.is_cache_valid", return_value=True), \
             patch("app.pipelines.scanner.price_changed", return_value=False):
            ticker, result, status = _process_one(
                "AAPL", old, force_refresh=False, price_threshold_pct=2.0
            )
        assert status == "cached"
        assert result is old

    def test_returns_failed_when_analyze_returns_none(self):
        from app.pipelines.scanner import _process_one
        with patch("app.pipelines.scanner.analyze_single_ticker", return_value=None):
            _, result, status = _process_one(
                "AAPL", None, force_refresh=True, price_threshold_pct=2.0
            )
        assert status == "failed"

    def test_returns_refreshed_when_cache_miss(self):
        from app.pipelines.scanner import _process_one
        new_data = _entry("AAPL")
        with patch("app.pipelines.scanner.analyze_single_ticker", return_value=new_data):
            _, result, status = _process_one(
                "AAPL", None, force_refresh=False, price_threshold_pct=2.0
            )
        assert status == "refreshed"
        assert result is new_data  # no blend when old is None

    def test_blends_when_old_exists_and_new_fetched(self):
        from app.pipelines.scanner import _process_one
        old = _entry("AAPL", composite_score=2.0)
        new = _entry("AAPL", composite_score=4.0)
        with patch("app.pipelines.scanner.is_cache_valid", return_value=False), \
             patch("app.pipelines.scanner.analyze_single_ticker", return_value=new), \
             patch("app.pipelines.scanner.dp_blend",
                   return_value={**new, "blended": True}) as mock_blend:
            _, result, status = _process_one(
                "AAPL", old, force_refresh=False, price_threshold_pct=2.0
            )
        assert result.get("blended") is True

    def test_force_refresh_bypasses_cache_validity(self):
        from app.pipelines.scanner import _process_one
        old = _entry("AAPL", hours_old=1.0)  # 유효한 캐시
        new = _entry("AAPL")
        with patch("app.pipelines.scanner.analyze_single_ticker", return_value=new):
            _, _, status = _process_one("AAPL", old, force_refresh=True, price_threshold_pct=2.0)
        assert status == "refreshed"

    def test_period_days_mismatch_forces_refresh(self):
        from app.pipelines.scanner import _process_one
        old = _entry("AAPL", hours_old=1.0, period_days=200)  # period가 다름
        new = _entry("AAPL")
        with patch("app.pipelines.scanner.is_cache_valid", return_value=True), \
             patch("app.pipelines.scanner.price_changed", return_value=False), \
             patch("app.pipelines.scanner.analyze_single_ticker", return_value=new):
            _, _, status = _process_one("AAPL", old, force_refresh=False,
                                         price_threshold_pct=2.0, period_days=400)
        assert status == "refreshed"

    def test_cache_valid_but_price_changed_triggers_refresh(self):
        from app.pipelines.scanner import _process_one
        old = _entry("AAPL", hours_old=1.0)
        new = _entry("AAPL")
        with patch("app.pipelines.scanner.is_cache_valid", return_value=True), \
             patch("app.pipelines.scanner.price_changed", return_value=True), \
             patch("app.pipelines.scanner.analyze_single_ticker", return_value=new):
            _, _, status = _process_one("AAPL", old, force_refresh=False, price_threshold_pct=2.0)
        assert status == "refreshed"


# ─────────────────────────────────────────────────────────────
# get_cache_stats
# ─────────────────────────────────────────────────────────────

class TestGetCacheStats:
    def test_counts_valid_stale_uncached(self):
        from app.pipelines.scanner import get_cache_stats
        cache = {
            "AAPL": _entry("AAPL", hours_old=1.0),   # valid
            "MSFT": _entry("MSFT", hours_old=25.0),  # stale
            # GOOG: uncached
        }
        with patch("app.pipelines.scanner.load_all_cache", return_value=cache), \
             patch("app.pipelines.scanner.is_cache_valid",
                   side_effect=lambda e, **_: e.get("ticker") == "AAPL"):
            stats = get_cache_stats(["AAPL", "MSFT", "GOOG"])

        assert stats["valid"] == 1
        assert stats["stale"] == 1
        assert stats["uncached"] == 1
        assert stats["total_cached"] == 2

    def test_all_cached_and_valid(self):
        from app.pipelines.scanner import get_cache_stats
        cache = {
            "AAPL": _entry("AAPL", hours_old=1.0),
            "MSFT": _entry("MSFT", hours_old=1.0),
        }
        with patch("app.pipelines.scanner.load_all_cache", return_value=cache), \
             patch("app.pipelines.scanner.is_cache_valid", return_value=True):
            stats = get_cache_stats(["AAPL", "MSFT"])

        assert stats["valid"] == 2
        assert stats["stale"] == 0
        assert stats["uncached"] == 0

    def test_all_uncached(self):
        from app.pipelines.scanner import get_cache_stats
        with patch("app.pipelines.scanner.load_all_cache", return_value={}):
            stats = get_cache_stats(["AAPL", "MSFT", "GOOG"])

        assert stats["total_cached"] == 0
        assert stats["uncached"] == 3
        assert stats["valid"] == 0

    def test_returns_ttl_hours_in_stats(self):
        from app.pipelines.scanner import CACHE_TTL_H, get_cache_stats
        with patch("app.pipelines.scanner.load_all_cache", return_value={}):
            stats = get_cache_stats([])
        assert stats["ttl_hours"] == CACHE_TTL_H


# ─────────────────────────────────────────────────────────────
# analyze_single_ticker
# ─────────────────────────────────────────────────────────────

class TestAnalyzeSingleTicker:
    def _make_full_df(self, n: int = 100) -> pd.DataFrame:
        close = np.linspace(100.0, 150.0, n)
        df = pd.DataFrame({
            "Open": close - 1, "High": close + 2, "Low": close - 2,
            "Close": close, "Volume": np.full(n, 1e6),
        })
        df["RSI14"] = 50.0
        df["BB_Position"] = 0.5
        df["ATR_Pct"] = 1.0
        df["MA5_vs_MA20"] = 0.01
        df["MA20_vs_MA50"] = 0.005
        df["Momentum_Normalized"] = 0.02
        df["MACD_Cross"] = 0
        return df

    def _make_mock_pred(self) -> MagicMock:
        pred = MagicMock()
        pred.train.return_value = {"model_type": "XGBoost", "cv_accuracy_mean": 0.55}
        pred.predict.return_value = {"signal": "BUY", "up_probability": 0.62}
        return pred

    # analyze_single_ticker 내부에서 local import 사용:
    #   from .fetcher import fetch_stock_data          → app.pipelines.fetcher
    #   from .technical import add_all_indicators, get_current_signals → app.pipelines.technical
    #   from ..models.predictor import EnsemblePredictor → app.models.predictor
    # 따라서 patch 경로는 원본 모듈을 대상으로 한다.

    def _patches(self, df, info, pred, signals):
        return [
            patch("app.pipelines.fetcher.fetch_stock_data", return_value=(df, info)),
            patch("app.pipelines.technical.add_all_indicators", return_value=df),
            patch("app.models.predictor.EnsemblePredictor", return_value=pred),
            patch("app.pipelines.technical.get_current_signals", return_value=signals),
        ]

    def test_returns_none_when_fetch_returns_none(self):
        from app.pipelines.scanner import analyze_single_ticker
        with patch("app.pipelines.fetcher.fetch_stock_data", return_value=(None, None)):
            result = analyze_single_ticker("AAPL")
        assert result is None

    def test_returns_none_when_df_too_short(self):
        from app.pipelines.scanner import analyze_single_ticker
        short_df = self._make_full_df(30)  # < 60행
        with patch("app.pipelines.fetcher.fetch_stock_data", return_value=(short_df, {})):
            result = analyze_single_ticker("AAPL")
        assert result is None

    def test_returns_none_when_train_returns_error(self):
        from app.pipelines.scanner import analyze_single_ticker
        df = self._make_full_df(100)
        mock_pred = MagicMock()
        mock_pred.train.return_value = {"error": "학습 데이터 부족"}
        with patch("app.pipelines.fetcher.fetch_stock_data", return_value=(df, {})), \
             patch("app.pipelines.technical.add_all_indicators", return_value=df), \
             patch("app.models.predictor.EnsemblePredictor", return_value=mock_pred):
            result = analyze_single_ticker("AAPL")
        assert result is None

    def test_returns_dict_with_required_fields_on_success(self):
        from app.pipelines.scanner import analyze_single_ticker
        df = self._make_full_df(100)
        mock_pred = self._make_mock_pred()
        signals = {"RSI": ("BUY", 50.0, "desc"), "MA": ("SELL", 60.0, "desc")}
        with patch("app.pipelines.fetcher.fetch_stock_data",
                   return_value=(df, {"shortName": "Apple"})), \
             patch("app.pipelines.technical.add_all_indicators", return_value=df), \
             patch("app.models.predictor.EnsemblePredictor", return_value=mock_pred), \
             patch("app.pipelines.technical.get_current_signals", return_value=signals):
            result = analyze_single_ticker("AAPL")

        assert result is not None
        required = {
            "ticker", "current_price", "up_probability", "estimated_upside",
            "composite_score", "rsi", "bb_position", "buy_signals", "sell_signals",
            "ml_signal", "cached_at", "analyzed_at", "dp_blend_count",
        }
        assert required.issubset(result.keys())

    def test_ticker_field_matches_input(self):
        from app.pipelines.scanner import analyze_single_ticker
        df = self._make_full_df(100)
        mock_pred = self._make_mock_pred()
        with patch("app.pipelines.fetcher.fetch_stock_data", return_value=(df, {})), \
             patch("app.pipelines.technical.add_all_indicators", return_value=df), \
             patch("app.models.predictor.EnsemblePredictor", return_value=mock_pred), \
             patch("app.pipelines.technical.get_current_signals", return_value={}):
            result = analyze_single_ticker("TSLA")
        assert result["ticker"] == "TSLA"

    def test_buy_sell_signal_counts_correct(self):
        from app.pipelines.scanner import analyze_single_ticker
        df = self._make_full_df(100)
        mock_pred = self._make_mock_pred()
        signals = {
            "RSI": ("BUY", 50.0, ""),
            "MA": ("BUY", 60.0, ""),
            "BB": ("SELL", 40.0, ""),
            "Vol": ("HOLD", 50.0, ""),
        }
        with patch("app.pipelines.fetcher.fetch_stock_data", return_value=(df, {})), \
             patch("app.pipelines.technical.add_all_indicators", return_value=df), \
             patch("app.models.predictor.EnsemblePredictor", return_value=mock_pred), \
             patch("app.pipelines.technical.get_current_signals", return_value=signals):
            result = analyze_single_ticker("AAPL")
        assert result["buy_signals"] == 2
        assert result["sell_signals"] == 1

    def test_returns_none_on_unexpected_exception(self):
        from app.pipelines.scanner import analyze_single_ticker
        with patch("app.pipelines.fetcher.fetch_stock_data", side_effect=RuntimeError("boom")):
            result = analyze_single_ticker("AAPL")
        assert result is None

    def test_dp_blend_count_initialized_to_one(self):
        from app.pipelines.scanner import analyze_single_ticker
        df = self._make_full_df(100)
        mock_pred = self._make_mock_pred()
        with patch("app.pipelines.fetcher.fetch_stock_data", return_value=(df, {})), \
             patch("app.pipelines.technical.add_all_indicators", return_value=df), \
             patch("app.models.predictor.EnsemblePredictor", return_value=mock_pred), \
             patch("app.pipelines.technical.get_current_signals", return_value={}):
            result = analyze_single_ticker("AAPL")
        assert result["dp_blend_count"] == 1


# ─────────────────────────────────────────────────────────────
# run_sp500_scan
# ─────────────────────────────────────────────────────────────

class TestRunSP500Scan:
    def _make_entries(self, tickers: list[str]) -> dict:
        return {t: _entry(t, composite_score=float(i + 1)) for i, t in enumerate(tickers)}

    def test_returns_dataframe_and_cache_dict(self):
        from app.pipelines.scanner import run_sp500_scan
        tickers = ["AAPL", "MSFT", "GOOG"]
        cache = self._make_entries(tickers)
        with patch("app.pipelines.scanner.load_all_cache", return_value=cache), \
             patch("app.pipelines.scanner.is_cache_valid", return_value=True), \
             patch("app.pipelines.scanner.price_changed", return_value=False), \
             patch("app.pipelines.scanner.save_all_cache"):
            df, returned_cache = run_sp500_scan(tickers)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(returned_cache, dict)

    def test_df_sorted_by_composite_score_descending(self):
        from app.pipelines.scanner import run_sp500_scan
        tickers = ["AAPL", "MSFT", "GOOG"]
        cache = {
            "AAPL": _entry("AAPL", composite_score=1.0),
            "MSFT": _entry("MSFT", composite_score=3.0),
            "GOOG": _entry("GOOG", composite_score=2.0),
        }
        with patch("app.pipelines.scanner.load_all_cache", return_value=cache), \
             patch("app.pipelines.scanner.is_cache_valid", return_value=True), \
             patch("app.pipelines.scanner.price_changed", return_value=False), \
             patch("app.pipelines.scanner.save_all_cache"):
            df, _ = run_sp500_scan(tickers)
        assert list(df["ticker"]) == ["MSFT", "GOOG", "AAPL"]

    def test_returns_empty_df_when_no_results(self):
        from app.pipelines.scanner import run_sp500_scan
        with patch("app.pipelines.scanner.load_all_cache", return_value={}), \
             patch("app.pipelines.scanner.analyze_single_ticker", return_value=None), \
             patch("app.pipelines.scanner.save_all_cache"):
            df, cache = run_sp500_scan(["AAPL"])
        assert df.empty

    def test_stop_flag_aborts_scan(self):
        from app.pipelines.scanner import run_sp500_scan
        tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "NVDA"]
        stop_flag = {"stop": True}
        with patch("app.pipelines.scanner.load_all_cache", return_value={}), \
             patch("app.pipelines.scanner.analyze_single_ticker", return_value=None), \
             patch("app.pipelines.scanner.save_all_cache"):
            df, _ = run_sp500_scan(tickers, stop_flag=stop_flag)
        # 중단이 발생해도 예외 없이 완료되어야 함
        assert isinstance(df, pd.DataFrame)

    def test_progress_object_is_updated(self):
        from app.pipelines.scanner import ScanProgress, run_sp500_scan
        tickers = ["AAPL", "MSFT"]
        cache = self._make_entries(tickers)
        progress = ScanProgress(total=len(tickers))
        with patch("app.pipelines.scanner.load_all_cache", return_value=cache), \
             patch("app.pipelines.scanner.is_cache_valid", return_value=True), \
             patch("app.pipelines.scanner.price_changed", return_value=False), \
             patch("app.pipelines.scanner.save_all_cache"):
            run_sp500_scan(tickers, progress=progress)
        assert progress.done == len(tickers)

    def test_progress_callback_is_called(self):
        from app.pipelines.scanner import ScanProgress, run_sp500_scan
        tickers = ["AAPL"]
        cache = self._make_entries(tickers)
        callback = MagicMock()
        progress = ScanProgress(total=1)
        with patch("app.pipelines.scanner.load_all_cache", return_value=cache), \
             patch("app.pipelines.scanner.is_cache_valid", return_value=True), \
             patch("app.pipelines.scanner.price_changed", return_value=False), \
             patch("app.pipelines.scanner.save_all_cache"):
            run_sp500_scan(tickers, progress=progress, progress_callback=callback)
        callback.assert_called()

    def test_df_index_starts_at_one(self):
        from app.pipelines.scanner import run_sp500_scan
        tickers = ["AAPL", "MSFT"]
        cache = self._make_entries(tickers)
        with patch("app.pipelines.scanner.load_all_cache", return_value=cache), \
             patch("app.pipelines.scanner.is_cache_valid", return_value=True), \
             patch("app.pipelines.scanner.price_changed", return_value=False), \
             patch("app.pipelines.scanner.save_all_cache"):
            df, _ = run_sp500_scan(tickers)
        assert df.index[0] == 1

    def test_cached_results_not_reanalyzed(self):
        from app.pipelines.scanner import run_sp500_scan
        tickers = ["AAPL"]
        cache = self._make_entries(tickers)
        with patch("app.pipelines.scanner.load_all_cache", return_value=cache), \
             patch("app.pipelines.scanner.is_cache_valid", return_value=True), \
             patch("app.pipelines.scanner.price_changed", return_value=False), \
             patch("app.pipelines.scanner.analyze_single_ticker") as mock_analyze, \
             patch("app.pipelines.scanner.save_all_cache"):
            run_sp500_scan(tickers)
        mock_analyze.assert_not_called()