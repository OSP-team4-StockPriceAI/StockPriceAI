"""
sentiment_store.py 단위 테스트

네트워크 없이 실행되도록 httpx를 전부 mock 처리.
테스트 대상 함수:
  - _group_by_date
  - save_sentiment_to_backend
  - save_sentiment_to_backend_async
  - purge_old_sentiments
  - load_sentiment_history
  - merge_sentiment_into_df
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.models.sentiment_store import (
    _group_by_date,
    load_sentiment_history,
    merge_sentiment_into_df,
    purge_old_sentiments,
    save_sentiment_to_backend,
    save_sentiment_to_backend_async,
)


# ─────────────────────────────────────────────────────────────
# 픽스처 / 헬퍼
# ─────────────────────────────────────────────────────────────

def _make_news_df(
    dates: list[str],
    compounds: list[float] | None = None,
    impact_scores: list[float] | None = None,
    labels: list[str] | None = None,
    hours_ago: list[float] | None = None,
) -> pd.DataFrame:
    """analyze_news_sentiment()가 반환하는 news_df 형태 더미 생성."""
    n = len(dates)
    return pd.DataFrame({
        "published_at": pd.to_datetime(dates),
        "compound":      compounds    if compounds    is not None else [0.5] * n,
        "impact_score":  impact_scores if impact_scores is not None else [1.0] * n,
        "label":         labels       if labels       is not None else ["POSITIVE"] * n,
        "hours_ago":     hours_ago    if hours_ago    is not None else [1.0] * n,
    })


def _make_ohlcv_df(dates: list[str]) -> pd.DataFrame:
    """Date 컬럼을 가진 더미 OHLCV DataFrame."""
    n = len(dates)
    close = np.linspace(100.0, 110.0, n)
    return pd.DataFrame({
        "Date":   dates,
        "Open":   close - 1,
        "High":   close + 2,
        "Low":    close - 2,
        "Close":  close,
        "Volume": np.full(n, 1_000_000, dtype=float),
    })


def _make_history_response(dates: list[str], scores: list[float]) -> dict:
    """Backend /history API 응답 형태 더미 생성."""
    return {
        "ticker": "AAPL",
        "count": len(dates),
        "history": [
            {
                "date": d,
                "avg_sentiment": s,
                "time_weighted_avg": s,
                "raw_avg": s,
                "impact_score_avg": s,
                "news_count": 3,
                "positive_pct": 60.0,
                "negative_pct": 20.0,
                "neutral_pct": 20.0,
                "signal": "BULLISH" if s > 0.1 else "BEARISH" if s < -0.1 else "NEUTRAL",
            }
            for d, s in zip(dates, scores)
        ],
    }


# ─────────────────────────────────────────────────────────────
# _group_by_date
# ─────────────────────────────────────────────────────────────

class TestGroupByDate:
    def test_empty_df_returns_empty_list(self):
        result = _group_by_date(pd.DataFrame(), "AAPL")
        assert result == []

    def test_single_date_single_article(self):
        news_df = _make_news_df(
            dates=["2024-01-01"],
            compounds=[0.5],
            impact_scores=[1.0],
            labels=["POSITIVE"],
            hours_ago=[1.0],
        )
        result = _group_by_date(news_df, "AAPL")

        assert len(result) == 1
        assert result[0]["date"] == "2024-01-01"
        assert result[0]["ticker"] == "AAPL"
        assert result[0]["news_count"] == 1
        assert result[0]["positive_pct"] == 100.0
        assert result[0]["negative_pct"] == 0.0
        assert result[0]["signal"] == "BULLISH"

    def test_multiple_dates_grouped_correctly(self):
        news_df = _make_news_df(
            dates=["2024-01-01", "2024-01-01", "2024-01-02"],
            compounds=[0.5, 0.3, -0.4],
            impact_scores=[1.0, 1.0, 1.0],
            labels=["POSITIVE", "POSITIVE", "NEGATIVE"],
        )
        result = _group_by_date(news_df, "AAPL")

        assert len(result) == 2
        dates = [r["date"] for r in result]
        assert "2024-01-01" in dates
        assert "2024-01-02" in dates

        day1 = next(r for r in result if r["date"] == "2024-01-01")
        assert day1["news_count"] == 2
        assert day1["positive_pct"] == 100.0

        day2 = next(r for r in result if r["date"] == "2024-01-02")
        assert day2["news_count"] == 1
        assert day2["negative_pct"] == 100.0

    def test_result_is_sorted_by_date_ascending(self):
        news_df = _make_news_df(
            dates=["2024-01-03", "2024-01-01", "2024-01-02"],
        )
        result = _group_by_date(news_df, "AAPL")

        assert [r["date"] for r in result] == ["2024-01-01", "2024-01-02", "2024-01-03"]

    def test_ticker_is_uppercased(self):
        news_df = _make_news_df(dates=["2024-01-01"])
        result = _group_by_date(news_df, "aapl")

        assert result[0]["ticker"] == "AAPL"

    def test_signal_bullish_when_avg_sentiment_above_threshold(self):
        news_df = _make_news_df(
            dates=["2024-01-01"],
            compounds=[0.8],
            impact_scores=[1.0],
        )
        result = _group_by_date(news_df, "AAPL")
        assert result[0]["signal"] == "BULLISH"

    def test_signal_bearish_when_avg_sentiment_below_threshold(self):
        news_df = _make_news_df(
            dates=["2024-01-01"],
            compounds=[-0.8],
            impact_scores=[-1.0],
            labels=["NEGATIVE"],
        )
        result = _group_by_date(news_df, "AAPL")
        assert result[0]["signal"] == "BEARISH"

    def test_signal_neutral_when_avg_sentiment_near_zero(self):
        news_df = _make_news_df(
            dates=["2024-01-01"],
            compounds=[0.05],
            impact_scores=[0.05],
            labels=["NEUTRAL"],
        )
        result = _group_by_date(news_df, "AAPL")
        assert result[0]["signal"] == "NEUTRAL"

    def test_avg_sentiment_clipped_to_minus_one_to_one(self):
        news_df = _make_news_df(
            dates=["2024-01-01"],
            compounds=[1.0],
            impact_scores=[99.0],  # 극단적으로 큰 값
        )
        result = _group_by_date(news_df, "AAPL")
        assert -1.0 <= result[0]["avg_sentiment"] <= 1.0

    def test_hours_ago_nan_uses_fallback_36(self):
        """hours_ago NaN이면 36시간으로 대체되어 계산이 깨지지 않아야 함."""
        news_df = _make_news_df(
            dates=["2024-01-01"],
            compounds=[0.5],
            hours_ago=[float("nan")],
        )
        result = _group_by_date(news_df, "AAPL")
        assert len(result) == 1
        assert result[0]["avg_sentiment"] is not None

    def test_pct_fields_sum_to_100(self):
        news_df = _make_news_df(
            dates=["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"],
            labels=["POSITIVE", "NEGATIVE", "NEUTRAL", "POSITIVE"],
        )
        result = _group_by_date(news_df, "AAPL")
        r = result[0]
        total = r["positive_pct"] + r["negative_pct"] + r["neutral_pct"]
        assert abs(total - 100.0) < 0.01

    def test_required_keys_present(self):
        news_df = _make_news_df(dates=["2024-01-01"])
        result = _group_by_date(news_df, "AAPL")
        expected_keys = {
            "ticker", "date", "avg_sentiment", "time_weighted_avg",
            "raw_avg", "impact_score_avg", "news_count",
            "positive_pct", "negative_pct", "neutral_pct", "signal",
        }
        assert expected_keys <= set(result[0].keys())


# ─────────────────────────────────────────────────────────────
# save_sentiment_to_backend (동기)
# ─────────────────────────────────────────────────────────────

class TestSaveSentimentToBackend:
    def test_empty_news_df_returns_zero_without_http_call(self):
        with patch("app.models.sentiment_store.httpx.post") as mock_post:
            result = save_sentiment_to_backend("AAPL", pd.DataFrame())

        mock_post.assert_not_called()
        assert result == {"saved": 0, "skipped": 0}

    def test_successful_save_returns_saved_skipped_counts(self):
        news_df = _make_news_df(dates=["2024-01-01", "2024-01-02"])
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"saved": 2, "skipped": 0}
        mock_resp.raise_for_status = MagicMock()

        with patch("app.models.sentiment_store.httpx.post", return_value=mock_resp) as mock_post:
            result = save_sentiment_to_backend("AAPL", news_df)

        mock_post.assert_called_once()
        assert result == {"saved": 2, "skipped": 0}

    def test_payload_ticker_is_uppercased(self):
        news_df = _make_news_df(dates=["2024-01-01"])
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"saved": 1, "skipped": 0}
        mock_resp.raise_for_status = MagicMock()

        with patch("app.models.sentiment_store.httpx.post", return_value=mock_resp) as mock_post:
            save_sentiment_to_backend("aapl", news_df)

        payload = mock_post.call_args.kwargs["json"]
        assert payload["ticker"] == "AAPL"

    def test_payload_items_count_matches_unique_dates(self):
        news_df = _make_news_df(
            dates=["2024-01-01", "2024-01-01", "2024-01-02"],
        )
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"saved": 2, "skipped": 0}
        mock_resp.raise_for_status = MagicMock()

        with patch("app.models.sentiment_store.httpx.post", return_value=mock_resp) as mock_post:
            save_sentiment_to_backend("AAPL", news_df)

        payload = mock_post.call_args.kwargs["json"]
        assert len(payload["items"]) == 2  # 날짜 2개로 그룹핑

    def test_http_error_returns_zero_without_raising(self):
        news_df = _make_news_df(dates=["2024-01-01"])

        with patch(
            "app.models.sentiment_store.httpx.post",
            side_effect=Exception("connection refused"),
        ):
            result = save_sentiment_to_backend("AAPL", news_df)

        assert result == {"saved": 0, "skipped": 0}

    def test_skipped_count_returned_from_backend(self):
        news_df = _make_news_df(dates=["2024-01-01"])
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"saved": 0, "skipped": 1}
        mock_resp.raise_for_status = MagicMock()

        with patch("app.models.sentiment_store.httpx.post", return_value=mock_resp):
            result = save_sentiment_to_backend("AAPL", news_df)

        assert result == {"saved": 0, "skipped": 1}

    def test_correct_url_called(self):
        news_df = _make_news_df(dates=["2024-01-01"])
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"saved": 1, "skipped": 0}
        mock_resp.raise_for_status = MagicMock()

        with patch("app.models.sentiment_store.httpx.post", return_value=mock_resp) as mock_post:
            with patch("app.models.sentiment_store.settings") as mock_settings:
                mock_settings.backend_url = "http://backend:8000"
                save_sentiment_to_backend("AAPL", news_df)

        called_url = mock_post.call_args.args[0]
        assert called_url == "http://backend:8000/api/v1/sentiment/batch"


# ─────────────────────────────────────────────────────────────
# save_sentiment_to_backend_async (비동기)
# ─────────────────────────────────────────────────────────────

class TestSaveSentimentToBackendAsync:
    @pytest.mark.asyncio
    async def test_empty_news_df_returns_zero_without_http_call(self):
        with patch("app.models.sentiment_store.httpx.AsyncClient") as mock_client:
            result = await save_sentiment_to_backend_async("AAPL", pd.DataFrame())

        mock_client.assert_not_called()
        assert result == {"saved": 0, "skipped": 0}

    @pytest.mark.asyncio
    async def test_successful_async_save(self):
        news_df = _make_news_df(dates=["2024-01-01"])

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"saved": 1, "skipped": 0}
        mock_resp.raise_for_status = MagicMock()

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_resp
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("app.models.sentiment_store.httpx.AsyncClient", return_value=mock_client_instance):
            result = await save_sentiment_to_backend_async("AAPL", news_df)

        assert result == {"saved": 1, "skipped": 0}

    @pytest.mark.asyncio
    async def test_http_error_returns_zero_without_raising(self):
        news_df = _make_news_df(dates=["2024-01-01"])

        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = Exception("timeout")
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("app.models.sentiment_store.httpx.AsyncClient", return_value=mock_client_instance):
            result = await save_sentiment_to_backend_async("AAPL", news_df)

        assert result == {"saved": 0, "skipped": 0}

    @pytest.mark.asyncio
    async def test_resp_json_called_inside_async_context(self):
        """resp.json()이 async with 블록 안에서 호출되는지 확인 (버그 방지)."""
        news_df = _make_news_df(dates=["2024-01-01"])

        call_order = []

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = lambda: call_order.append("json") or {"saved": 1, "skipped": 0}

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_resp
        mock_client_instance.__aenter__ = AsyncMock(
            side_effect=lambda: call_order.append("enter") or mock_client_instance
        )
        mock_client_instance.__aexit__ = AsyncMock(
            side_effect=lambda *a: call_order.append("exit")
        )

        with patch("app.models.sentiment_store.httpx.AsyncClient", return_value=mock_client_instance):
            await save_sentiment_to_backend_async("AAPL", news_df)

        # json()은 반드시 exit(컨텍스트 종료) 이전에 호출되어야 함
        assert "json" in call_order
        assert call_order.index("json") < call_order.index("exit")


# ─────────────────────────────────────────────────────────────
# purge_old_sentiments
# ─────────────────────────────────────────────────────────────

class TestPurgeOldSentiments:
    def test_successful_purge_returns_deleted_count(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"deleted": 5, "cutoff_date": "2023-01-01"}
        mock_resp.raise_for_status = MagicMock()

        with patch("app.models.sentiment_store.httpx.delete", return_value=mock_resp):
            result = purge_old_sentiments("AAPL", period_days=365)

        assert result == 5

    def test_correct_url_and_params_sent(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"deleted": 0}
        mock_resp.raise_for_status = MagicMock()

        with patch("app.models.sentiment_store.httpx.delete", return_value=mock_resp) as mock_del:
            with patch("app.models.sentiment_store.settings") as mock_settings:
                mock_settings.backend_url = "http://backend:8000"
                purge_old_sentiments("AAPL", period_days=400)

        called_url = mock_del.call_args.args[0]
        called_params = mock_del.call_args.kwargs["params"]
        assert called_url == "http://backend:8000/api/v1/sentiment/AAPL/purge"
        assert called_params == {"period_days": 400}

    def test_ticker_uppercased_in_url(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"deleted": 0}
        mock_resp.raise_for_status = MagicMock()

        with patch("app.models.sentiment_store.httpx.delete", return_value=mock_resp) as mock_del:
            with patch("app.models.sentiment_store.settings") as mock_settings:
                mock_settings.backend_url = "http://backend:8000"
                purge_old_sentiments("aapl", period_days=365)

        called_url = mock_del.call_args.args[0]
        assert "AAPL" in called_url

    def test_http_error_returns_zero_without_raising(self):
        with patch(
            "app.models.sentiment_store.httpx.delete",
            side_effect=Exception("connection error"),
        ):
            result = purge_old_sentiments("AAPL", period_days=365)

        assert result == 0

    def test_zero_deleted_returns_zero(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"deleted": 0}
        mock_resp.raise_for_status = MagicMock()

        with patch("app.models.sentiment_store.httpx.delete", return_value=mock_resp):
            result = purge_old_sentiments("AAPL", period_days=365)

        assert result == 0


# ─────────────────────────────────────────────────────────────
# load_sentiment_history
# ─────────────────────────────────────────────────────────────

class TestLoadSentimentHistory:
    def test_returns_dataframe_with_sentiment_columns(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_history_response(
            ["2024-01-01", "2024-01-02"], [0.3, -0.2]
        )
        mock_resp.raise_for_status = MagicMock()

        with patch("app.models.sentiment_store.httpx.get", return_value=mock_resp):
            df = load_sentiment_history("AAPL")

        assert not df.empty
        assert "Sentiment_Score" in df.columns
        assert "Sentiment_Positive" in df.columns
        assert "Sentiment_Negative" in df.columns

    def test_sentiment_score_equals_avg_sentiment(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_history_response(["2024-01-01"], [0.4])
        mock_resp.raise_for_status = MagicMock()

        with patch("app.models.sentiment_store.httpx.get", return_value=mock_resp):
            df = load_sentiment_history("AAPL")

        assert df["Sentiment_Score"].iloc[0] == pytest.approx(0.4)

    def test_sentiment_positive_clips_negative_scores_to_zero(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_history_response(["2024-01-01"], [-0.5])
        mock_resp.raise_for_status = MagicMock()

        with patch("app.models.sentiment_store.httpx.get", return_value=mock_resp):
            df = load_sentiment_history("AAPL")

        assert df["Sentiment_Positive"].iloc[0] == 0.0
        assert df["Sentiment_Negative"].iloc[0] == pytest.approx(0.5)

    def test_empty_history_returns_empty_dataframe(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ticker": "AAPL", "count": 0, "history": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("app.models.sentiment_store.httpx.get", return_value=mock_resp):
            df = load_sentiment_history("AAPL")

        assert df.empty

    def test_http_error_returns_empty_dataframe(self):
        with patch(
            "app.models.sentiment_store.httpx.get",
            side_effect=Exception("timeout"),
        ):
            df = load_sentiment_history("AAPL")

        assert df.empty

    def test_correct_url_and_limit_param_sent(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"history": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("app.models.sentiment_store.httpx.get", return_value=mock_resp) as mock_get:
            with patch("app.models.sentiment_store.settings") as mock_settings:
                mock_settings.backend_url = "http://backend:8000"
                load_sentiment_history("AAPL", limit=200)

        called_url = mock_get.call_args.args[0]
        called_params = mock_get.call_args.kwargs["params"]
        assert called_url == "http://backend:8000/api/v1/sentiment/AAPL/history"
        assert called_params == {"limit": 200}

    def test_ticker_uppercased_in_url(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"history": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("app.models.sentiment_store.httpx.get", return_value=mock_resp) as mock_get:
            with patch("app.models.sentiment_store.settings") as mock_settings:
                mock_settings.backend_url = "http://backend:8000"
                load_sentiment_history("aapl")

        called_url = mock_get.call_args.args[0]
        assert "AAPL" in called_url


# ─────────────────────────────────────────────────────────────
# merge_sentiment_into_df
# ─────────────────────────────────────────────────────────────

class TestMergeSentimentIntoDf:
    def _mock_load(self, dates: list[str], scores: list[float]):
        """load_sentiment_history를 원하는 DataFrame으로 패치."""
        sent_df = pd.DataFrame({
            "date":              dates,
            "avg_sentiment":     scores,
            "Sentiment_Score":   scores,
            "Sentiment_Positive": [max(0.0, s) for s in scores],
            "Sentiment_Negative": [max(0.0, -s) for s in scores],
        })
        return patch(
            "app.models.sentiment_store.load_sentiment_history",
            return_value=sent_df,
        )

    def _mock_empty_load(self):
        return patch(
            "app.models.sentiment_store.load_sentiment_history",
            return_value=pd.DataFrame(),
        )

    def test_sentiment_columns_added_to_df(self):
        ohlcv = _make_ohlcv_df(["2024-01-01", "2024-01-02"])
        with self._mock_load(["2024-01-01", "2024-01-02"], [0.3, -0.2]):
            result = merge_sentiment_into_df(ohlcv, "AAPL")

        assert "Sentiment_Score" in result.columns
        assert "Sentiment_Positive" in result.columns
        assert "Sentiment_Negative" in result.columns
        assert "Sentiment_Missing" in result.columns

    def test_matched_dates_have_correct_sentiment_score(self):
        ohlcv = _make_ohlcv_df(["2024-01-01", "2024-01-02"])
        with self._mock_load(["2024-01-01", "2024-01-02"], [0.3, -0.2]):
            result = merge_sentiment_into_df(ohlcv, "AAPL")

        assert result.loc[result["Date"] == "2024-01-01", "Sentiment_Score"].iloc[0] == pytest.approx(0.3)
        assert result.loc[result["Date"] == "2024-01-02", "Sentiment_Score"].iloc[0] == pytest.approx(-0.2)

    def test_unmatched_dates_filled_with_zero(self):
        ohlcv = _make_ohlcv_df(["2024-01-01", "2024-01-03"])  # 01-03은 DB에 없음
        with self._mock_load(["2024-01-01"], [0.5]):
            result = merge_sentiment_into_df(ohlcv, "AAPL")

        assert result.loc[result["Date"] == "2024-01-03", "Sentiment_Score"].iloc[0] == 0.0
        # 뉴스 없는 날은 Sentiment_Missing=1
        assert result.loc[result["Date"] == "2024-01-03", "Sentiment_Missing"].iloc[0] == 1.0
        # 뉴스 있는 날은 Sentiment_Missing=0
        assert result.loc[result["Date"] == "2024-01-01", "Sentiment_Missing"].iloc[0] == 0.0

    def test_empty_sentiment_history_fills_all_zeros(self):
        ohlcv = _make_ohlcv_df(["2024-01-01", "2024-01-02"])
        with self._mock_empty_load():
            result = merge_sentiment_into_df(ohlcv, "AAPL")

        assert (result["Sentiment_Score"] == 0.0).all()
        assert (result["Sentiment_Positive"] == 0.0).all()
        assert (result["Sentiment_Negative"] == 0.0).all()
        assert (result["Sentiment_Missing"] == 1.0).all()

    def test_row_count_preserved_after_merge(self):
        ohlcv = _make_ohlcv_df(["2024-01-01", "2024-01-02", "2024-01-03"])
        with self._mock_load(["2024-01-01"], [0.5]):
            result = merge_sentiment_into_df(ohlcv, "AAPL")

        assert len(result) == 3

    def test_datetime_index_df_handled(self):
        """DatetimeIndex(index.name=None)를 가진 df도 KeyError 없이 merge 가능해야 함.

        버그 케이스: reset_index() 시 index.name이 None이면 컬럼명이 "index"가 되어
        df["Date"] KeyError 발생 → index.strftime()으로 직접 추출하도록 수정됨.
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        close = np.linspace(100.0, 103.0, 3)
        ohlcv = pd.DataFrame({"Close": close}, index=dates)

        # 전제: Date 컬럼 없고, index.name은 None, DatetimeIndex
        assert "Date" not in ohlcv.columns
        assert ohlcv.index.name is None
        assert isinstance(ohlcv.index, pd.DatetimeIndex)

        date_strs = dates.strftime("%Y-%m-%d").tolist()
        sent_df = pd.DataFrame({
            "date":               date_strs,
            "avg_sentiment":      [0.2, 0.3, 0.4],
            "Sentiment_Score":    [0.2, 0.3, 0.4],
            "Sentiment_Positive": [0.2, 0.3, 0.4],
            "Sentiment_Negative": [0.0, 0.0, 0.0],
        })
        with patch("app.models.sentiment_store.load_sentiment_history", return_value=sent_df):
            result = merge_sentiment_into_df(ohlcv, "AAPL")

        assert "Sentiment_Score" in result.columns
        assert "Sentiment_Missing" in result.columns
        assert len(result) == 3
        assert result["Sentiment_Score"].notna().all()
        assert (result["Sentiment_Score"] > 0).all()
        # 모든 날짜가 매칭됐으므로 Sentiment_Missing은 전부 0
        assert (result["Sentiment_Missing"] == 0.0).all()

    def test_no_date_column_or_index_fills_zeros(self):
        """Date 컬럼도 DatetimeIndex도 없으면 감정지수 0으로 채워야 함."""
        ohlcv = pd.DataFrame({"Close": [100.0, 101.0]})  # 정수 인덱스
        with self._mock_load(["2024-01-01"], [0.5]):
            result = merge_sentiment_into_df(ohlcv, "AAPL")

        assert (result["Sentiment_Score"] == 0.0).all()
        assert (result["Sentiment_Missing"] == 1.0).all()

    def test_original_df_not_mutated(self):
        ohlcv = _make_ohlcv_df(["2024-01-01", "2024-01-02"])
        original_cols = set(ohlcv.columns)

        with self._mock_load(["2024-01-01"], [0.5]):
            merge_sentiment_into_df(ohlcv, "AAPL")

        assert set(ohlcv.columns) == original_cols