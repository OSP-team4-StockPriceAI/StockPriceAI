"""
fetcher.py 단위 테스트

네트워크 없이 실행되도록 yfinance를 전부 mock 처리.
테스트 대상 함수:
  - is_korean_ticker
  - normalize_ticker
  - fetch_stock_data
  - fetch_earnings_history
  - fetch_institutional_holders
  - get_market_context
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────
# 픽스처 / 헬퍼
# ─────────────────────────────────────────────────────────────

def _make_price_df(n: int = 60, tz_aware: bool = False) -> pd.DataFrame:
    """더미 OHLCV DataFrame 생성."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    if tz_aware:
        idx = idx.tz_localize("America/New_York")
    close = np.linspace(100.0, 150.0, n)
    return pd.DataFrame(
        {
            "Open": close - 1,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": np.full(n, 1_000_000, dtype=float),
        },
        index=idx,
    )


def _make_mock_ticker(
    hist: pd.DataFrame | None = None,
    info: dict | None = None,
    quarterly_income_stmt: pd.DataFrame | None = None,
    quarterly_financials: pd.DataFrame | None = None,
    institutional_holders: pd.DataFrame | None = None,
) -> MagicMock:
    """yf.Ticker() 반환값 mock."""
    mock = MagicMock()
    mock.history.return_value = hist if hist is not None else _make_price_df()
    mock.info = info if info is not None else {"shortName": "Test Corp", "sector": "Technology"}
    mock.quarterly_income_stmt = quarterly_income_stmt
    mock.quarterly_financials = quarterly_financials
    mock.institutional_holders = institutional_holders
    return mock


# ─────────────────────────────────────────────────────────────
# is_korean_ticker
# ─────────────────────────────────────────────────────────────

class TestIsKoreanTicker:
    def test_six_digit_number_is_korean(self):
        from app.pipelines.fetcher import is_korean_ticker
        assert is_korean_ticker("005930") is True

    def test_ks_suffix_is_korean(self):
        from app.pipelines.fetcher import is_korean_ticker
        assert is_korean_ticker("005930.KS") is True

    def test_kq_suffix_is_korean(self):
        from app.pipelines.fetcher import is_korean_ticker
        assert is_korean_ticker("035720.KQ") is True

    def test_lowercase_suffix_is_korean(self):
        from app.pipelines.fetcher import is_korean_ticker
        assert is_korean_ticker("005930.ks") is True

    def test_us_ticker_is_not_korean(self):
        from app.pipelines.fetcher import is_korean_ticker
        assert is_korean_ticker("AAPL") is False

    def test_five_digit_is_not_korean(self):
        from app.pipelines.fetcher import is_korean_ticker
        assert is_korean_ticker("05930") is False

    def test_seven_digit_is_not_korean(self):
        from app.pipelines.fetcher import is_korean_ticker
        assert is_korean_ticker("0059301") is False

    def test_alphanumeric_is_not_korean(self):
        from app.pipelines.fetcher import is_korean_ticker
        assert is_korean_ticker("TSLA") is False


# ─────────────────────────────────────────────────────────────
# normalize_ticker
# ─────────────────────────────────────────────────────────────

class TestNormalizeTicker:
    def test_six_digit_appends_ks(self):
        from app.pipelines.fetcher import normalize_ticker
        assert normalize_ticker("005930") == "005930.KS"

    def test_already_ks_unchanged(self):
        from app.pipelines.fetcher import normalize_ticker
        assert normalize_ticker("005930.KS") == "005930.KS"

    def test_us_ticker_uppercased(self):
        from app.pipelines.fetcher import normalize_ticker
        assert normalize_ticker("aapl") == "AAPL"

    def test_strips_whitespace(self):
        from app.pipelines.fetcher import normalize_ticker
        assert normalize_ticker("  AAPL  ") == "AAPL"

    def test_kq_suffix_preserved(self):
        from app.pipelines.fetcher import normalize_ticker
        assert normalize_ticker("035720.KQ") == "035720.KQ"


# ─────────────────────────────────────────────────────────────
# fetch_stock_data
# ─────────────────────────────────────────────────────────────

class TestFetchStockData:
    @patch("app.pipelines.fetcher.redis.from_url")
    def test_returns_dataframe_and_info_on_success(self, mock_redis):
        # mock redis to return None (cache miss)
        mock_redis.return_value.get.return_value = None
        
        from app.pipelines.fetcher import fetch_stock_data

        mock_ticker = _make_mock_ticker(
            hist=_make_price_df(60),
            info={"shortName": "Apple Inc.", "sector": "Technology", "trailingPE": 28.5},
        )
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=mock_ticker):
            df, info = fetch_stock_data("AAPL", period_days=365)

        assert df is not None and info is not None
        assert len(df) == 60
        # Verify cache was saved
        mock_redis.return_value.setex.assert_called_once()

    @patch("app.pipelines.fetcher.redis.from_url")
    def test_ohlcv_columns_present(self, mock_redis):
        mock_redis.return_value.get.return_value = None
        from app.pipelines.fetcher import fetch_stock_data

        with patch("app.pipelines.fetcher.yf.Ticker", return_value=_make_mock_ticker()):
            df, _ = fetch_stock_data("AAPL")

        assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(df.columns)

    @patch("app.pipelines.fetcher.redis.from_url")
    def test_columns_are_float32(self, mock_redis):
        mock_redis.return_value.get.return_value = None
        from app.pipelines.fetcher import fetch_stock_data

        with patch("app.pipelines.fetcher.yf.Ticker", return_value=_make_mock_ticker()):
            df, _ = fetch_stock_data("AAPL")

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert df[col].dtype == np.float32, f"{col} should be float32"

    @patch("app.pipelines.fetcher.redis.from_url")
    def test_timezone_aware_index_is_converted_to_naive(self, mock_redis):
        mock_redis.return_value.get.return_value = None
        from app.pipelines.fetcher import fetch_stock_data

        tz_hist = _make_price_df(60, tz_aware=True)
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=_make_mock_ticker(hist=tz_hist)):
            df, _ = fetch_stock_data("AAPL")

        assert df.index.tz is None

    @patch("app.pipelines.fetcher.redis.from_url")
    def test_index_is_datetime(self, mock_redis):
        mock_redis.return_value.get.return_value = None
        from app.pipelines.fetcher import fetch_stock_data

        with patch("app.pipelines.fetcher.yf.Ticker", return_value=_make_mock_ticker()):
            df, _ = fetch_stock_data("AAPL")

        assert isinstance(df.index, pd.DatetimeIndex)

    @patch("app.pipelines.fetcher.redis.from_url")
    def test_empty_history_returns_none_tuple(self, mock_redis):
        mock_redis.return_value.get.return_value = None
        from app.pipelines.fetcher import fetch_stock_data

        mock_ticker = _make_mock_ticker(hist=pd.DataFrame())
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=mock_ticker):
            df, info = fetch_stock_data("AAPL")

        assert df is None and info is None

    @patch("app.pipelines.fetcher.redis.from_url")
    def test_short_history_under_30_rows_returns_none(self, mock_redis):
        mock_redis.return_value.get.return_value = None
        from app.pipelines.fetcher import fetch_stock_data

        mock_ticker = _make_mock_ticker(hist=_make_price_df(10))
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=mock_ticker):
            df, info = fetch_stock_data("AAPL")

        assert df is None and info is None

    @patch("app.pipelines.fetcher.redis.from_url")
    def test_yfinance_exception_raises_value_error(self, mock_redis):
        mock_redis.return_value.get.return_value = None
        from app.pipelines.fetcher import fetch_stock_data

        with patch("app.pipelines.fetcher.yf.Ticker", side_effect=Exception("network error")):
            with pytest.raises(ValueError, match="데이터 수집 실패"):
                fetch_stock_data("AAPL")

    @patch("app.pipelines.fetcher.redis.from_url")
    def test_period_zero_uses_max(self, mock_redis):
        """period_days=0 이면 history(period='max') 호출해야 함."""
        mock_redis.return_value.get.return_value = None
        from app.pipelines.fetcher import fetch_stock_data

        mock_ticker = _make_mock_ticker(hist=_make_price_df(60))
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=mock_ticker):
            fetch_stock_data("AAPL", period_days=0)

        mock_ticker.history.assert_called_once_with(period="max", auto_adjust=True)

    @patch("app.pipelines.fetcher.redis.from_url")
    def test_period_positive_uses_start_end(self, mock_redis):
        """period_days>0 이면 start/end 파라미터로 history 호출해야 함."""
        mock_redis.return_value.get.return_value = None
        from app.pipelines.fetcher import fetch_stock_data

        mock_ticker = _make_mock_ticker(hist=_make_price_df(60))
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=mock_ticker):
            fetch_stock_data("AAPL", period_days=365)

        call_kwargs = mock_ticker.history.call_args.kwargs
        assert "start" in call_kwargs and "end" in call_kwargs

    @patch("app.pipelines.fetcher.redis.from_url")
    def test_korean_ticker_normalized_before_request(self, mock_redis):
        """6자리 숫자 종목코드가 .KS 형식으로 변환되어 yf.Ticker에 전달되어야 함."""
        mock_redis.return_value.get.return_value = None
        from app.pipelines.fetcher import fetch_stock_data

        mock_ticker = _make_mock_ticker(hist=_make_price_df(60))
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=mock_ticker) as mock_yf:
            fetch_stock_data("005930")

        mock_yf.assert_called_once_with("005930.KS")

    @patch("app.pipelines.fetcher.redis.from_url")
    def test_financial_info_keys_extracted(self, mock_redis):
        mock_redis.return_value.get.return_value = None
        from app.pipelines.fetcher import fetch_stock_data

        raw_info = {
            "trailingPE": 28.5,
            "priceToBook": 3.2,
            "sector": "Technology",
            "shortName": "Apple Inc.",
            "unknownKey": "should_not_appear",
        }
        mock_ticker = _make_mock_ticker(hist=_make_price_df(60), info=raw_info)
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=mock_ticker):
            _, info = fetch_stock_data("AAPL")

        assert info["trailingPE"] == 28.5
        assert info["sector"] == "Technology"
        assert "unknownKey" not in info

    @patch("app.pipelines.fetcher.redis.from_url")
    def test_info_fetch_failure_returns_empty_info(self, mock_redis):
        """stock.info 접근 시 예외가 발생해도 price_df는 정상 반환해야 함."""
        mock_redis.return_value.get.return_value = None
        from app.pipelines.fetcher import fetch_stock_data

        mock_ticker = _make_mock_ticker(hist=_make_price_df(60))
        type(mock_ticker).info = property(
            lambda self: (_ for _ in ()).throw(Exception("info error"))
        )

        with patch("app.pipelines.fetcher.yf.Ticker", return_value=mock_ticker):
            df, info = fetch_stock_data("AAPL")

        assert df is not None
        assert info == {}

    @patch("app.pipelines.fetcher.redis.from_url")
    def test_no_none_values_in_returned_info(self, mock_redis):
        """None 값을 가진 재무 항목은 info dict에서 제외되어야 함."""
        mock_redis.return_value.get.return_value = None
        from app.pipelines.fetcher import fetch_stock_data

        raw_info = {"trailingPE": 28.5, "forwardPE": None, "sector": "Technology"}
        mock_ticker = _make_mock_ticker(hist=_make_price_df(60), info=raw_info)
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=mock_ticker):
            _, info = fetch_stock_data("AAPL")

        assert "forwardPE" not in info

    @patch("app.pipelines.fetcher.redis.from_url")
    def test_redis_cache_hit_returns_data_without_yfinance(self, mock_redis):
        """Redis 캐시에 데이터가 있으면 yfinance를 호출하지 않고 반환해야 함."""
        import json
        
        # Mock Redis return value
        cached_json = json.dumps({
            "info": {"shortName": "Apple", "trailingPE": 28.5},
            "history": [
                {
                    "Date": "2024-01-01T00:00:00",
                    "Open": 100, "High": 105, "Low": 99, "Close": 104, "Volume": 1000,
                }
            ]
        })
        mock_redis.return_value.get.return_value = cached_json
        
        from app.pipelines.fetcher import fetch_stock_data
        
        # yf.Ticker should not be called
        with patch("app.pipelines.fetcher.yf.Ticker") as mock_yf:
            df, info = fetch_stock_data("AAPL")
            
            mock_yf.assert_not_called()
            
            assert df is not None
            assert len(df) == 1
            assert df.index.name == "Date"
            assert "Close" in df.columns
            assert df.iloc[0]["Close"] == 104.0
            
            assert info is not None
            assert info["shortName"] == "Apple"

    @patch("app.pipelines.fetcher.redis.from_url")
    def test_redis_cache_exception_falls_back_to_yfinance(self, mock_redis):
        """Redis 연결/조회 중 예외 발생 시 yfinance로 폴백해야 함."""
        mock_redis.return_value.get.side_effect = Exception("Redis connection error")
        
        from app.pipelines.fetcher import fetch_stock_data
        
        mock_ticker = _make_mock_ticker(hist=_make_price_df(60))
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=mock_ticker) as mock_yf:
            df, info = fetch_stock_data("AAPL")
            
            # yfinance must be called
            mock_yf.assert_called_once()
            assert df is not None
            assert len(df) == 60


# ─────────────────────────────────────────────────────────────
# fetch_earnings_history
# ─────────────────────────────────────────────────────────────

class TestFetchEarningsHistory:
    def _make_income_stmt(self) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=4, freq="QE")
        return pd.DataFrame(
            {
                "Total Revenue": [100e9, 95e9, 90e9, 85e9],
                "Net Income": [25e9, 23e9, 22e9, 20e9],
                "Gross Profit": [45e9, 43e9, 40e9, 38e9],
            },
            index=idx,
        ).T  # yfinance 형식: 컬럼=날짜, 인덱스=항목

    def test_returns_dataframe_from_income_stmt(self):
        from app.pipelines.fetcher import fetch_earnings_history

        stmt = self._make_income_stmt()
        mock_ticker = _make_mock_ticker(quarterly_income_stmt=stmt)
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=mock_ticker):
            result = fetch_earnings_history("AAPL")

        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_falls_back_to_quarterly_financials(self):
        """quarterly_income_stmt가 없으면 quarterly_financials를 사용해야 함."""
        from app.pipelines.fetcher import fetch_earnings_history

        fallback = self._make_income_stmt()
        mock_ticker = _make_mock_ticker(
            quarterly_income_stmt=pd.DataFrame(),  # 비어있는 stmt
            quarterly_financials=fallback,
        )
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=mock_ticker):
            result = fetch_earnings_history("AAPL")

        assert result is not None

    def test_returns_none_when_all_sources_empty(self):
        from app.pipelines.fetcher import fetch_earnings_history

        mock_ticker = _make_mock_ticker(
            quarterly_income_stmt=pd.DataFrame(),
            quarterly_financials=pd.DataFrame(),
        )
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=mock_ticker):
            result = fetch_earnings_history("AAPL")

        assert result is None

    def test_returns_none_on_exception(self):
        from app.pipelines.fetcher import fetch_earnings_history

        with patch("app.pipelines.fetcher.yf.Ticker", side_effect=Exception("error")):
            result = fetch_earnings_history("AAPL")

        assert result is None

    def test_result_rows_capped_at_8(self):
        from app.pipelines.fetcher import fetch_earnings_history

        idx = pd.date_range("2020-01-01", periods=12, freq="QE")
        long_stmt = pd.DataFrame(
            {"Total Revenue": range(12), "Net Income": range(12)}, index=idx
        ).T
        mock_ticker = _make_mock_ticker(quarterly_income_stmt=long_stmt)
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=mock_ticker):
            result = fetch_earnings_history("AAPL")

        assert result is not None and len(result) <= 8

    def test_korean_ticker_normalized(self):
        from app.pipelines.fetcher import fetch_earnings_history

        stmt = self._make_income_stmt()
        mock_ticker = _make_mock_ticker(quarterly_income_stmt=stmt)
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=mock_ticker) as mock_yf:
            fetch_earnings_history("005930")

        mock_yf.assert_called_once_with("005930.KS")


# ─────────────────────────────────────────────────────────────
# fetch_institutional_holders
# ─────────────────────────────────────────────────────────────

class TestFetchInstitutionalHolders:
    def _make_holders_df(self, n: int = 5) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Holder": [f"Institution {i}" for i in range(n)],
                "Shares": [1_000_000 * (n - i) for i in range(n)],
                "% Out": [round(0.1 * (n - i), 2) for i in range(n)],
            }
        )

    def test_returns_dataframe_on_success(self):
        from app.pipelines.fetcher import fetch_institutional_holders

        mock_ticker = _make_mock_ticker(institutional_holders=self._make_holders_df(5))
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=mock_ticker):
            result = fetch_institutional_holders("AAPL")

        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_result_capped_at_10_rows(self):
        from app.pipelines.fetcher import fetch_institutional_holders

        mock_ticker = _make_mock_ticker(institutional_holders=self._make_holders_df(15))
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=mock_ticker):
            result = fetch_institutional_holders("AAPL")

        assert result is not None and len(result) <= 10

    def test_returns_none_when_holders_empty(self):
        from app.pipelines.fetcher import fetch_institutional_holders

        mock_ticker = _make_mock_ticker(institutional_holders=pd.DataFrame())
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=mock_ticker):
            result = fetch_institutional_holders("AAPL")

        assert result is None

    def test_returns_none_when_holders_is_none(self):
        from app.pipelines.fetcher import fetch_institutional_holders

        mock_ticker = _make_mock_ticker(institutional_holders=None)
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=mock_ticker):
            result = fetch_institutional_holders("AAPL")

        assert result is None

    def test_returns_none_on_exception(self):
        from app.pipelines.fetcher import fetch_institutional_holders

        with patch("app.pipelines.fetcher.yf.Ticker", side_effect=Exception("error")):
            result = fetch_institutional_holders("AAPL")

        assert result is None


# ─────────────────────────────────────────────────────────────
# get_market_context
# ─────────────────────────────────────────────────────────────

class TestGetMarketContext:
    def _make_bench_ticker(
        self, start: float = 100.0, end: float = 110.0, n: int = 60
    ) -> MagicMock:
        hist = pd.DataFrame(
            {"Close": np.linspace(start, end, n)},
            index=pd.date_range("2024-01-01", periods=n, freq="B"),
        )
        mock = MagicMock()
        mock.history.return_value = hist
        return mock

    def test_us_ticker_uses_sp500_benchmark(self):
        from app.pipelines.fetcher import get_market_context

        bench = self._make_bench_ticker()
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=bench) as mock_yf:
            get_market_context("AAPL")

        mock_yf.assert_called_once_with("^GSPC")

    def test_ks_ticker_uses_kospi_benchmark(self):
        from app.pipelines.fetcher import get_market_context

        bench = self._make_bench_ticker()
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=bench) as mock_yf:
            get_market_context("005930.KS")

        mock_yf.assert_called_once_with("^KS11")

    def test_kq_ticker_uses_kospi_benchmark(self):
        from app.pipelines.fetcher import get_market_context

        bench = self._make_bench_ticker()
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=bench) as mock_yf:
            get_market_context("035720.KQ")

        mock_yf.assert_called_once_with("^KS11")

    def test_returns_benchmark_name_and_return(self):
        from app.pipelines.fetcher import get_market_context

        with patch(
            "app.pipelines.fetcher.yf.Ticker",
            return_value=self._make_bench_ticker(100, 110),
        ):
            context = get_market_context("AAPL")

        assert context["benchmark_name"] == "S&P 500"
        assert "benchmark_3mo_return" in context

    def test_return_pct_calculated_correctly(self):
        from app.pipelines.fetcher import get_market_context

        # 100 → 110: 수익률 10%
        with patch(
            "app.pipelines.fetcher.yf.Ticker",
            return_value=self._make_bench_ticker(100, 110),
        ):
            context = get_market_context("AAPL")

        assert abs(context["benchmark_3mo_return"] - 10.0) < 0.1

    def test_benchmark_fetch_failure_returns_empty_dict(self):
        from app.pipelines.fetcher import get_market_context

        with patch("app.pipelines.fetcher.yf.Ticker", side_effect=Exception("network error")):
            context = get_market_context("AAPL")

        assert context == {}

    def test_empty_benchmark_history_returns_empty_dict(self):
        from app.pipelines.fetcher import get_market_context

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        with patch("app.pipelines.fetcher.yf.Ticker", return_value=mock_ticker):
            context = get_market_context("AAPL")

        assert context == {}

    def test_return_value_is_rounded_to_two_decimals(self):
        from app.pipelines.fetcher import get_market_context

        with patch(
            "app.pipelines.fetcher.yf.Ticker",
            return_value=self._make_bench_ticker(100, 107.777),
        ):
            context = get_market_context("AAPL")

        ret = context.get("benchmark_3mo_return", 0)
        assert ret == round(ret, 2)

# ─────────────────────────────────────────────────────────────
# get_recent_SP500_tickers.py 테스트
# ─────────────────────────────────────────────────────────────

from unittest.mock import MagicMock, patch  # noqa: E402


class TestSP500Tickers:
    def test_normalize_ticker(self):
        from app.pipelines.get_recent_SP500_tickers import _normalize_ticker
        assert _normalize_ticker("BRK.B") == "BRK-B"
        assert _normalize_ticker("AAPL") == "AAPL"

    def test_fetch_from_wikipedia_network_failure(self):
        """네트워크 실패 시 빈 리스트 반환."""
        from app.pipelines.get_recent_SP500_tickers import _fetch_from_wikipedia
        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            result = _fetch_from_wikipedia()
        assert result == []

    def test_get_sp500_tickers_uses_fallback_when_fetch_fails(self):
        from app.pipelines.get_recent_SP500_tickers import get_sp500_tickers
        get_sp500_tickers.cache_clear()
        with patch("app.pipelines.get_recent_SP500_tickers._fetch_from_wikipedia", return_value=[]):
            result = get_sp500_tickers()
        assert len(result) > 0
        assert "AAPL" in result
        get_sp500_tickers.cache_clear()

    def test_get_sp500_tickers_deduplicates(self):
        from app.pipelines.get_recent_SP500_tickers import get_sp500_tickers
        get_sp500_tickers.cache_clear()
        with patch("app.pipelines.get_recent_SP500_tickers._fetch_from_wikipedia",
                   return_value=["AAPL", "MSFT", "AAPL"]):
            result = get_sp500_tickers()
        assert result.count("AAPL") == 1
        get_sp500_tickers.cache_clear()

    def test_refresh_sp500_tickers(self):
        from app.pipelines.get_recent_SP500_tickers import refresh_sp500_tickers
        with patch("app.pipelines.get_recent_SP500_tickers._fetch_from_wikipedia",
                   return_value=["TEST1", "TEST2"]):
            result = refresh_sp500_tickers()
        assert "TEST1" in result

    def test_fetch_from_wikipedia_no_symbol_column(self):
        """파싱했지만 Symbol 컬럼이 없을 때 빈 리스트."""
        from app.pipelines.get_recent_SP500_tickers import _fetch_from_wikipedia
        mock_df = pd.DataFrame({"Company": ["Apple"], "HQ": ["Cupertino"]})
        with patch("urllib.request.urlopen") as mock_open, \
             patch("pandas.read_html", return_value=[mock_df]):
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"<html></html>"
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = _fetch_from_wikipedia()
        assert result == []

    def test_fetch_from_wikipedia_success(self):
        """파싱 성공 시 티커 리스트 반환."""
        from app.pipelines.get_recent_SP500_tickers import _fetch_from_wikipedia
        mock_df = pd.DataFrame({"Symbol": ["AAPL", "MSFT", "BRK.B"]})
        with patch("urllib.request.urlopen") as mock_open, \
             patch("pandas.read_html", return_value=[mock_df]):
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"<html></html>"
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = _fetch_from_wikipedia()
        assert "AAPL" in result
        assert "BRK-B" in result  # 정규화됨

def test_fetch_stock_data_force_refresh_skips_cache():
    """force_refresh=True이면 캐시를 무시하고 yfinance로 수집한다."""
    from unittest.mock import MagicMock, patch
    import pandas as pd
    from app.pipelines.fetcher import fetch_stock_data

    n = 30
    mock_df = pd.DataFrame(
        {
            "Open": [100.0] * n,
            "High": [110.0] * n,
            "Low": [90.0] * n,
            "Close": [105.0] * n,
            "Volume": [1000] * n,
        },
        index=pd.date_range("2024-01-01", periods=n),
    )
    mock_df.index.name = "Date"

    with patch("app.pipelines.fetcher.yf.Ticker") as mock_ticker:
        inst = MagicMock()
        inst.history.return_value = mock_df
        inst.info = {"shortName": "Apple"}
        mock_ticker.return_value = inst
        df, info = fetch_stock_data("AAPL", force_refresh=True)

    assert df is not None