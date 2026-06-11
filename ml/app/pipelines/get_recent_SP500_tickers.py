"""
S&P 500 최신 종목 리스트 취득 모듈

Wikipedia의 S&P 500 종목 페이지를 파싱하여 최신 티커 목록을 반환합니다.
네트워크 오류나 파싱 실패 시 하드코딩된 폴백 리스트를 반환합니다.

사용 예시:
    from ml.app.pipelines.get_recent_SP500_tickers import get_sp500_tickers
    tickers = get_sp500_tickers()
"""

import logging
from functools import lru_cache

import pandas as pd

log = logging.getLogger("stockai.sp500")

# ─────────────────────────────────────────────────────────────
# 폴백용 하드코딩 리스트 (Wikipedia 접근 실패 시 사용)
# ─────────────────────────────────────────────────────────────
_FALLBACK_TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA", "AVGO", "META", "GOOGL", "GOOG", "TSLA", "ORCL", "CRM",
    "AMD", "QCOM", "TXN", "INTC", "ADI", "MU", "AMAT", "LRCX", "KLAC", "MRVL",
    "CDNS", "SNPS", "FTNT", "PANW", "CRWD", "NOW", "ADSK", "ANSS", "GDDY", "PAYC",
    "TTWO", "EA", "AKAM", "CTSH", "EPAM", "FFIV", "NTAP", "STX", "WDC",
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
    "GIS", "CPB", "SJM", "CAG", "HRL", "MKC", "CHD", "CLX", "EL",
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "DVN",
    "HAL", "BKR", "OXY", "APA",
    "CAT", "HON", "UPS", "BA", "RTX", "LMT", "NOC", "GD", "DE", "EMR",
    "ETN", "ROK", "AME", "FTV", "CARR", "OTIS", "TDG", "HWM", "GE",
    "WM", "RSG", "WCN", "CTAS", "PAYX", "ADP", "VRSK",
    "LIN", "APD", "SHW", "FCX", "NEM", "NUE", "STLD",
    "AMT", "PLD", "EQIX", "CCI", "SPG", "PSA", "EXR", "WELL", "VTR",
    "NEE", "DUK", "SO", "D", "EXC", "AEP", "SRE", "XEL", "WEC",
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR",
]


def _normalize_ticker(ticker: str) -> str:
    """yfinance 호환 형식으로 티커 변환 (예: BRK.B → BRK-B)."""
    return ticker.replace(".", "-")


def _fetch_from_wikipedia() -> list[str]:
    """
    Wikipedia의 S&P 500 종목 테이블을 파싱하여 티커 목록을 반환합니다.
    브라우저 User-Agent를 사용하여 403 차단을 우회합니다.
    실패 시 빈 리스트를 반환합니다.
    """
    import io
    import urllib.request

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8")

        tables = pd.read_html(io.StringIO(html), header=0)
        df = tables[0]

        # 'Symbol' 또는 'Ticker symbol' 컬럼 탐색
        col = next(
            (c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()),
            None,
        )
        if col is None:
            log.warning("Wikipedia 테이블에서 티커 컬럼을 찾지 못했습니다.")
            return []

        tickers = [_normalize_ticker(str(t).strip()) for t in df[col].dropna()]
        log.info(f"Wikipedia에서 S&P 500 종목 {len(tickers)}개 로드 완료")
        return tickers

    except Exception as e:
        log.warning(f"Wikipedia S&P 500 파싱 실패: {e}")
        return []


@lru_cache(maxsize=1)
def get_sp500_tickers(use_cache: bool = True) -> list[str]:
    """
    최신 S&P 500 티커 목록을 반환합니다.

    Args:
        use_cache: True면 프로세스 내 메모리 캐시를 사용합니다 (기본값).
                   False면 캐시를 무시하고 Wikipedia를 다시 조회합니다.

    Returns:
        yfinance 호환 티커 문자열 리스트 (중복 제거 완료)
    """
    tickers = _fetch_from_wikipedia()

    if not tickers:
        log.warning(f"폴백 리스트 사용 ({len(_FALLBACK_TICKERS)}개 종목)")
        tickers = _FALLBACK_TICKERS

    # 중복 제거, 순서 유지
    return list(dict.fromkeys(tickers))


def refresh_sp500_tickers() -> list[str]:
    """
    메모리 캐시를 초기화하고 Wikipedia에서 최신 종목 목록을 다시 조회합니다.

    Returns:
        최신 S&P 500 티커 리스트
    """
    get_sp500_tickers.cache_clear()
    return get_sp500_tickers()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tickers = get_sp500_tickers()
    print(f"총 {len(tickers)}개 S&P 500 종목:")
    print(", ".join(tickers[:20]), "...")
    sys.exit(0)
