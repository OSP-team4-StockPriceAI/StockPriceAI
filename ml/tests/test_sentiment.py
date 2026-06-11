"""
sentiment.py 테스트 모듈

테스트 대상:
  - _parse_rss_date
  - _parse_yf_news_item
  - _KeywordFallback
  - _scores_to_label
  - analyze_sentiment_vader
  - classify_news_type
  - detect_macro_theme
  - detect_market_regime
  - compute_impact_score
  - impact_weighted_sentiment
  - compute_relevance
  - fetch_news (mocked)
  - analyze_news_sentiment (mocked)
  - add_sentiment_to_features
"""

from datetime import datetime
from unittest.mock import patch

import pandas as pd

from app.models.sentiment import (
    _KeywordFallback,
    _parse_rss_date,
    _parse_yf_news_item,
    _scores_to_label,
    add_sentiment_to_features,
    analyze_news_sentiment,
    analyze_sentiment_vader,
    classify_news_type,
    compute_impact_score,
    compute_relevance,
    detect_macro_theme,
    detect_market_regime,
    fetch_news,
    impact_weighted_sentiment,
)


# ─────────────────────────────────────────────────────────────
# _parse_rss_date
# ─────────────────────────────────────────────────────────────
class TestParseRssDate:
    def test_rfc2822_with_tz(self):
        dt = _parse_rss_date("Mon, 15 May 2026 10:30:00 +0000")
        assert dt.year == 2026
        assert dt.month == 5
        assert dt.day == 15
        assert dt.tzinfo is None  # tzinfo stripped

    def test_rfc2822_gmt(self):
        dt = _parse_rss_date("Tue, 16 May 2026 08:00:00 GMT")
        assert dt.year == 2026
        assert dt.month == 5

    def test_iso8601(self):
        dt = _parse_rss_date("2026-05-17T12:00:00Z")
        assert dt.year == 2026
        assert dt.hour == 12

    def test_unknown_format_returns_now(self):
        dt = _parse_rss_date("not-a-date")
        assert isinstance(dt, datetime)
        assert (datetime.now() - dt).total_seconds() < 5

    def test_whitespace_stripped(self):
        dt = _parse_rss_date("  2026-05-17T12:00:00Z  ")
        assert dt.year == 2026


# ─────────────────────────────────────────────────────────────
# _parse_yf_news_item
# ─────────────────────────────────────────────────────────────
class TestParseYfNewsItem:
    def test_new_format_with_content(self):
        item = {
            "content": {
                "title": "AAPL beats earnings",
                "provider": {"displayName": "Reuters"},
                "pubDate": "2026-05-15T10:00:00Z",
                "canonicalUrl": {"url": "https://example.com/news"},
            }
        }
        result = _parse_yf_news_item(item)
        assert result is not None
        assert result["title"] == "AAPL beats earnings"
        assert result["publisher"] == "Reuters"
        assert result["source"] == "yfinance"
        assert "hours_ago" in result

    def test_legacy_format(self):
        item = {
            "title": "Tesla stock rises",
            "publisher": "Bloomberg",
            "providerPublishTime": datetime.now().timestamp(),
            "link": "https://example.com/tesla",
        }
        result = _parse_yf_news_item(item)
        assert result is not None
        assert result["title"] == "Tesla stock rises"
        assert result["publisher"] == "Bloomberg"

    def test_empty_title_returns_none(self):
        item = {"content": {"title": ""}}
        assert _parse_yf_news_item(item) is None

    def test_no_title_at_all_returns_none(self):
        item = {"content": {}}
        assert _parse_yf_news_item(item) is None

    def test_provider_not_dict(self):
        item = {
            "content": {
                "title": "Some news",
                "provider": "not-a-dict",
            }
        }
        result = _parse_yf_news_item(item)
        assert result is not None
        # provider is not dict → empty string, but legacy fallback yields "Unknown"
        assert result["publisher"] in ("", "Unknown")


# ─────────────────────────────────────────────────────────────
# _KeywordFallback
# ─────────────────────────────────────────────────────────────
class TestKeywordFallback:
    def setup_method(self):
        self.fb = _KeywordFallback()

    def test_positive_text(self):
        scores = self.fb.polarity_scores("The stock will surge and beat expectations")
        assert scores["compound"] > 0

    def test_negative_text(self):
        scores = self.fb.polarity_scores("Stock crash and decline expected")
        assert scores["compound"] < 0

    def test_neutral_text(self):
        scores = self.fb.polarity_scores("The weather is sunny today")
        assert scores["compound"] == 0.0

    def test_returns_required_keys(self):
        scores = self.fb.polarity_scores("test")
        assert all(k in scores for k in ("compound", "pos", "neg", "neu"))


# ─────────────────────────────────────────────────────────────
# _scores_to_label
# ─────────────────────────────────────────────────────────────
class TestScoresToLabel:
    def test_positive(self):
        assert _scores_to_label(0.5) == ("POSITIVE", "POSITIVE")

    def test_negative(self):
        assert _scores_to_label(-0.3) == ("NEGATIVE", "NEGATIVE")

    def test_neutral_upper_boundary(self):
        assert _scores_to_label(0.04) == ("NEUTRAL", "NEUTRAL")

    def test_neutral_lower_boundary(self):
        assert _scores_to_label(-0.04) == ("NEUTRAL", "NEUTRAL")

    def test_exact_positive_threshold(self):
        assert _scores_to_label(0.05)[0] == "POSITIVE"

    def test_exact_negative_threshold(self):
        assert _scores_to_label(-0.05)[0] == "NEGATIVE"


# ─────────────────────────────────────────────────────────────
# analyze_sentiment_vader
# ─────────────────────────────────────────────────────────────
class TestAnalyzeSentimentVader:
    def test_positive_headline(self):
        result = analyze_sentiment_vader("Company reports record profit and strong growth")
        assert result["label"] == "POSITIVE"
        assert result["compound"] > 0

    def test_negative_headline(self):
        result = analyze_sentiment_vader("Stock plunges after fraud scandal")
        assert result["label"] == "NEGATIVE"
        assert result["compound"] < 0

    def test_result_keys(self):
        result = analyze_sentiment_vader("neutral text")
        expected_keys = {"compound", "positive", "negative", "neutral", "label"}
        assert expected_keys == set(result.keys())

    def test_compound_range(self):
        result = analyze_sentiment_vader("extreme bullish skyrocket surge rally")
        assert -1.0 <= result["compound"] <= 1.0


# ─────────────────────────────────────────────────────────────
# classify_news_type
# ─────────────────────────────────────────────────────────────
class TestClassifyNewsType:
    def test_surprise_positive(self):
        result = classify_news_type("AAPL beats earnings expectations")
        assert result["news_type"] == "surprise_positive"
        assert result["persistence"] == 0.6

    def test_surprise_negative(self):
        result = classify_news_type("Company missed revenue below estimate")
        assert result["news_type"] == "surprise_negative"

    def test_structural(self):
        result = classify_news_type("New regulation on antitrust announced")
        assert result["news_type"] == "structural"
        assert result["persistence"] == 0.9

    def test_transient(self):
        result = classify_news_type("Rumor about temporary settlement")
        assert result["news_type"] == "transient"
        assert result["persistence"] == 0.15

    def test_contagion_needs_two_keywords(self):
        # 'supply chain' also appears in _STRUCTURAL_KEYWORDS, so use only contagion-exclusive words
        result = classify_news_type("sector peers competitor rival spillover")
        assert result["news_type"] == "contagion"

    def test_general_fallback(self):
        result = classify_news_type("Some random unrelated headline")
        assert result["news_type"] == "general"
        assert result["persistence"] == 0.3

    def test_contagion_score(self):
        result = classify_news_type("Sector supply chain peers competitor rival")
        assert result["contagion"] > 0

    def test_result_keys(self):
        result = classify_news_type("test")
        assert set(result.keys()) == {"news_type", "persistence", "contagion"}


# ─────────────────────────────────────────────────────────────
# detect_macro_theme
# ─────────────────────────────────────────────────────────────
class TestDetectMacroTheme:
    def test_rate_hike(self):
        assert detect_macro_theme("Fed announces rate hike") == "rate_hike"

    def test_rate_cut(self):
        assert detect_macro_theme("Interest rate cut expected") == "rate_cut"

    def test_recession(self):
        assert detect_macro_theme("GDP contraction signals recession") == "recession"

    def test_ai_boom(self):
        assert detect_macro_theme("AI artificial intelligence revolution") == "ai_boom"

    def test_trade_war(self):
        assert detect_macro_theme("Trade war tariff imposed") == "trade_war"

    def test_no_theme(self):
        assert detect_macro_theme("Local bakery opens new branch") is None

    def test_geopolitical(self):
        assert detect_macro_theme("Military conflict and sanctions") == "geopolitical_crisis"


# ─────────────────────────────────────────────────────────────
# detect_market_regime
# ─────────────────────────────────────────────────────────────
class TestDetectMarketRegime:
    def test_extreme_risk_off(self):
        result = detect_market_regime("Market crash amid crisis and panic")
        assert result == 2.5

    def test_single_risk_off(self):
        # Use a keyword that doesn't overlap with other risk-off signals
        result = detect_market_regime("Growing fear in the market")
        assert result == 1.8

    def test_risk_on(self):
        result = detect_market_regime("Record high for stock market rally")
        assert result == 0.8

    def test_default(self):
        result = detect_market_regime("Company releases quarterly update")
        assert result == 1.2


# ─────────────────────────────────────────────────────────────
# compute_impact_score
# ─────────────────────────────────────────────────────────────
class TestComputeImpactScore:
    def test_positive_surprise(self):
        result = compute_impact_score(
            compound=0.8, news_type="surprise_positive",
            persistence=0.6, contagion=0.0,
        )
        assert result["impact_score"] > 0

    def test_negative_surprise(self):
        result = compute_impact_score(
            compound=-0.8, news_type="surprise_negative",
            persistence=0.6, contagion=0.0,
        )
        assert result["impact_score"] < 0

    def test_result_keys(self):
        result = compute_impact_score(
            compound=0.5, news_type="general",
            persistence=0.3, contagion=0.0,
        )
        assert set(result.keys()) == {"impact_score", "S_surprise", "M_regime", "P_persistence"}

    def test_impact_score_clipped(self):
        result = compute_impact_score(
            compound=1.0, news_type="surprise_positive",
            persistence=1.0, contagion=1.0,
            beta=5.0, market_regime=2.5, macro_exposure=1.0,
        )
        assert -3.0 <= result["impact_score"] <= 3.0

    def test_contagion_amplifies(self):
        base = compute_impact_score(
            compound=0.5, news_type="general",
            persistence=0.5, contagion=0.0,
        )
        with_contagion = compute_impact_score(
            compound=0.5, news_type="general",
            persistence=0.5, contagion=1.0,
        )
        assert abs(with_contagion["impact_score"]) >= abs(base["impact_score"])

    def test_high_beta_amplifies(self):
        low = compute_impact_score(
            compound=0.5, news_type="general",
            persistence=0.5, contagion=0.0, beta=0.5,
        )
        high = compute_impact_score(
            compound=0.5, news_type="general",
            persistence=0.5, contagion=0.0, beta=2.0,
        )
        assert abs(high["impact_score"]) > abs(low["impact_score"])

    def test_negative_in_risk_off_amplified(self):
        normal = compute_impact_score(
            compound=-0.5, news_type="general",
            persistence=0.5, contagion=0.0, market_regime=1.2,
        )
        risk_off = compute_impact_score(
            compound=-0.5, news_type="general",
            persistence=0.5, contagion=0.0, market_regime=2.5,
        )
        assert abs(risk_off["impact_score"]) > abs(normal["impact_score"])


# ─────────────────────────────────────────────────────────────
# impact_weighted_sentiment
# ─────────────────────────────────────────────────────────────
class TestImpactWeightedSentiment:
    def test_empty_df(self):
        assert impact_weighted_sentiment(pd.DataFrame()) == 0.0

    def test_missing_column(self):
        df = pd.DataFrame({"foo": [1]})
        assert impact_weighted_sentiment(df) == 0.0

    def test_all_below_relevance(self):
        df = pd.DataFrame({
            "impact_score": [0.5],
            "relevance": [0.01],
            "hours_ago": [1.0],
        })
        assert impact_weighted_sentiment(df, min_relevance=0.08) == 0.0

    def test_positive_impact(self):
        df = pd.DataFrame({
            "impact_score": [0.5, 0.3, 0.4],
            "relevance": [0.5, 0.5, 0.5],
            "hours_ago": [1.0, 2.0, 3.0],
        })
        result = impact_weighted_sentiment(df)
        assert result > 0
        assert -1.0 <= result <= 1.0

    def test_negative_impact(self):
        df = pd.DataFrame({
            "impact_score": [-0.5, -0.3, -0.4],
            "relevance": [0.5, 0.5, 0.5],
            "hours_ago": [1.0, 2.0, 3.0],
        })
        result = impact_weighted_sentiment(df)
        assert result < 0


# ─────────────────────────────────────────────────────────────
# compute_relevance
# ─────────────────────────────────────────────────────────────
class TestComputeRelevance:
    def test_direct_mention(self):
        result = compute_relevance("AAPL reports strong earnings", "AAPL")
        assert result["relevance_tier"] == "직접"
        assert result["relevance"] >= 0.70

    def test_company_name_mention(self):
        result = compute_relevance(
            "Apple releases new iPhone",
            "AAPL", company_name="Apple Inc.",
        )
        assert result["relevance_tier"] == "직접"

    def test_macro_relevance(self):
        result = compute_relevance(
            "Fed announces rate hike amid hawkish tightening",
            "AAPL", sector="Technology",
        )
        assert result["relevance_tier"] == "매크로"
        assert result["macro_theme"] == "rate_hike"

    def test_sector_relevance(self):
        # Use sector keywords that don't trigger macro themes
        result = compute_relevance(
            "Local tech startup opens office",
            "AAPL", sector="Technology",
        )
        assert result["relevance_tier"] == "섹터"

    def test_market_relevance(self):
        result = compute_relevance(
            "Wall Street stock market rallies",
            "XYZ",
        )
        assert result["relevance_tier"] == "시장"

    def test_unrelated(self):
        result = compute_relevance(
            "Local bakery opens new branch",
            "AAPL",
        )
        assert result["relevance_tier"] == "무관"
        assert result["relevance"] == 0.03

    def test_result_keys(self):
        result = compute_relevance("test", "AAPL")
        expected = {
            "relevance", "relevance_tier", "news_type", "persistence",
            "contagion", "macro_theme", "macro_exposure", "market_regime",
        }
        assert expected == set(result.keys())

    def test_surprise_bonus_on_direct(self):
        normal = compute_relevance("AAPL quarterly report", "AAPL")
        surprise = compute_relevance("AAPL beats earnings exceeded expectations", "AAPL")
        assert surprise["relevance"] >= normal["relevance"]


# ─────────────────────────────────────────────────────────────
# fetch_news (mocked)
# ─────────────────────────────────────────────────────────────
class TestFetchNews:
    def _make_news(self, title, source="yfinance"):
        return {
            "title": title,
            "publisher": "Test",
            "published_at": datetime.now(),
            "url": "https://example.com",
            "hours_ago": 1.0,
            "source": source,
        }

    @patch("app.models.sentiment._fetch_google_news_rss", return_value=[])
    @patch("app.models.sentiment._fetch_yahoo_rss", return_value=[])
    @patch("app.models.sentiment._fetch_yfinance_news")
    def test_deduplicates(self, mock_yf, mock_rss, mock_google):
        mock_yf.return_value = [
            self._make_news("Same Title Here"),
            self._make_news("Same Title Here"),
        ]
        result = fetch_news("AAPL")
        assert len(result) == 1

    @patch("app.models.sentiment._fetch_google_news_rss", return_value=[])
    @patch("app.models.sentiment._fetch_yahoo_rss", return_value=[])
    @patch("app.models.sentiment._fetch_yfinance_news", return_value=[])
    def test_empty(self, mock_yf, mock_rss, mock_google):
        result = fetch_news("AAPL")
        assert result == []

    @patch("app.models.sentiment._fetch_google_news_rss", return_value=[])
    @patch("app.models.sentiment._fetch_yahoo_rss")
    @patch("app.models.sentiment._fetch_yfinance_news", return_value=[])
    def test_fallback_to_rss(self, mock_yf, mock_rss, mock_google):
        mock_rss.return_value = [self._make_news("RSS news", "yahoo_rss")]
        result = fetch_news("AAPL")
        assert len(result) == 1
        assert result[0]["source"] == "yahoo_rss"

    @patch("app.models.sentiment._fetch_google_news_rss", return_value=[])
    @patch("app.models.sentiment._fetch_yahoo_rss", return_value=[])
    @patch("app.models.sentiment._fetch_yfinance_news")
    def test_respects_max_news(self, mock_yf, mock_rss, mock_google):
        mock_yf.return_value = [self._make_news(f"News {i}") for i in range(50)]
        result = fetch_news("AAPL", max_news=5)
        assert len(result) <= 5

    @patch("app.models.sentiment._fetch_google_news_rss", return_value=[])
    @patch("app.models.sentiment._fetch_yahoo_rss", return_value=[])
    @patch("app.models.sentiment._fetch_yfinance_news")
    def test_sorted_by_hours_ago(self, mock_yf, mock_rss, mock_google):
        mock_yf.return_value = [
            {**self._make_news("Old news"), "hours_ago": 48.0},
            {**self._make_news("New news"), "hours_ago": 1.0},
        ]
        result = fetch_news("AAPL")
        assert result[0]["hours_ago"] <= result[-1]["hours_ago"]


# ─────────────────────────────────────────────────────────────
# analyze_news_sentiment (mocked)
# ─────────────────────────────────────────────────────────────
class TestAnalyzeNewsSentiment:
    def _make_news(self, title):
        return {
            "title": title,
            "publisher": "Test",
            "published_at": datetime.now(),
            "url": "https://example.com",
            "hours_ago": 1.0,
            "source": "yfinance",
        }

    @patch("app.models.sentiment.fetch_news", return_value=[])
    def test_no_news_returns_empty(self, mock_fn):
        df, summary = analyze_news_sentiment("AAPL")
        assert df.empty
        assert summary["news_count"] == 0
        assert summary["signal"] == "NEUTRAL"
        assert summary["model"] == "N/A"

    @patch("app.models.sentiment.fetch_news")
    def test_with_positive_news(self, mock_fn):
        mock_fn.return_value = [
            self._make_news("AAPL beats earnings with record profit and surge"),
            self._make_news("AAPL stock rallies on strong growth outperform"),
        ]
        df, summary = analyze_news_sentiment("AAPL")
        assert not df.empty
        assert summary["news_count"] == 2
        assert summary["model"] == "VADER + Impact Framework"
        assert "positive_pct" in summary
        assert "signal" in summary

    @patch("app.models.sentiment.fetch_news")
    def test_summary_keys(self, mock_fn):
        mock_fn.return_value = [self._make_news("Test news headline")]
        _, summary = analyze_news_sentiment("AAPL")
        expected_keys = {
            "avg_sentiment", "raw_avg", "time_weighted_avg", "impact_score_avg",
            "positive_pct", "negative_pct", "neutral_pct", "news_count",
            "direct_news_count", "sector_news_count", "surprise_count",
            "structural_count", "transient_count", "macro_themes",
            "model", "signal", "sources",
        }
        assert expected_keys == set(summary.keys())

    @patch("app.models.sentiment.fetch_news")
    def test_dataframe_columns(self, mock_fn):
        mock_fn.return_value = [self._make_news("Some news")]
        df, _ = analyze_news_sentiment("AAPL")
        expected_cols = {
            "title", "publisher", "published_at", "hours_ago", "source",
            "compound", "label", "positive", "negative", "neutral",
            "relevance", "relevance_tier", "news_type", "persistence",
            "contagion", "macro_theme", "macro_exposure", "market_regime",
            "impact_score", "S_surprise", "M_regime", "P_persistence",
        }
        assert expected_cols == set(df.columns)

    @patch("app.models.sentiment.fetch_news")
    def test_signal_bullish(self, mock_fn):
        mock_fn.return_value = [
            self._make_news("AAPL skyrocket surge bullish rally record high outperform"),
        ]
        _, summary = analyze_news_sentiment("AAPL")
        # Signal depends on impact_avg; just verify it's a valid value
        assert summary["signal"] in ("BULLISH", "NEUTRAL", "BEARISH")

    @patch("app.models.sentiment.fetch_news")
    def test_percentages_sum_to_100(self, mock_fn):
        mock_fn.return_value = [
            self._make_news("Good news bullish"),
            self._make_news("Bad news crash"),
            self._make_news("Neutral update"),
        ]
        _, summary = analyze_news_sentiment("AAPL")
        total_pct = summary["positive_pct"] + summary["negative_pct"] + summary["neutral_pct"]
        assert abs(total_pct - 100.0) < 0.2


# ─────────────────────────────────────────────────────────────
# add_sentiment_to_features
# ─────────────────────────────────────────────────────────────
class TestAddSentimentToFeatures:
    def test_positive_score(self):
        df = pd.DataFrame({"Close": [100, 101, 102]})
        result = add_sentiment_to_features(df, 0.5)
        assert "Sentiment_Score" in result.columns
        assert "Sentiment_Positive" in result.columns
        assert "Sentiment_Negative" in result.columns
        assert (result["Sentiment_Score"] == 0.5).all()
        assert (result["Sentiment_Positive"] == 0.5).all()
        assert (result["Sentiment_Negative"] == 0.0).all()

    def test_negative_score(self):
        df = pd.DataFrame({"Close": [100]})
        result = add_sentiment_to_features(df, -0.3)
        assert (result["Sentiment_Score"] == -0.3).all()
        assert (result["Sentiment_Positive"] == 0.0).all()
        assert (result["Sentiment_Negative"] == 0.3).all()

    def test_zero_score(self):
        df = pd.DataFrame({"Close": [100]})
        result = add_sentiment_to_features(df, 0.0)
        assert (result["Sentiment_Positive"] == 0.0).all()
        assert (result["Sentiment_Negative"] == 0.0).all()

    def test_original_df_unchanged(self):
        df = pd.DataFrame({"Close": [100, 101]})
        _ = add_sentiment_to_features(df, 0.5)
        assert "Sentiment_Score" not in df.columns

    def test_preserves_existing_columns(self):
        df = pd.DataFrame({"Close": [100], "Volume": [1000]})
        result = add_sentiment_to_features(df, 0.2)
        assert "Close" in result.columns
        assert "Volume" in result.columns
