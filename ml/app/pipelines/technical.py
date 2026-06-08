"""
Hybrid Indicator Engineering Module
"""


from typing import Any

import numpy as np
import pandas as pd


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    avg_gain = delta.clip(lower=0).ewm(com=window - 1, min_periods=window).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def calculate_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def calculate_bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = calculate_sma(series, window)
    std = series.rolling(window=window, min_periods=1).std()
    return middle + std * num_std, middle, middle - std * num_std


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    hi, lo, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(com=window - 1, min_periods=window).mean()


def calculate_directional_indicators(
    df: pd.DataFrame, window: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    hi, lo = df["High"], df["Low"]
    prev_hi, prev_lo = hi.shift(1), lo.shift(1)

    up_move = hi - prev_hi
    down_move = prev_lo - lo
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    plus_dm_smooth = plus_dm.ewm(com=window - 1, min_periods=window).mean()
    minus_dm_smooth = minus_dm.ewm(com=window - 1, min_periods=window).mean()
    atr = calculate_atr(df, window).replace(0, np.nan)

    plus_di = (100 * plus_dm_smooth / atr).fillna(0).astype("float32")
    minus_di = (100 * minus_dm_smooth / atr).fillna(0).astype("float32")
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)).fillna(0) * 100
    adx = dx.ewm(com=window - 1, min_periods=window).mean().fillna(0).astype("float32")

    return plus_di, minus_di, adx


def calculate_regime_probabilities(
    ma5: pd.Series,
    ma20: pd.Series,
    ma50: pd.Series,
    plus_di: pd.Series,
    minus_di: pd.Series,
    adx: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ma_bias_short = ((ma5 - ma20) / ma20).replace([np.inf, -np.inf], 0).fillna(0)
    ma_bias_mid = ((ma20 - ma50) / ma50).replace([np.inf, -np.inf], 0).fillna(0)
    alignment = ((ma_bias_short + ma_bias_mid) / 2).clip(-1, 1)

    directional = (
        (plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    ).fillna(0).clip(-1, 1)
    trend_strength = (adx / 100).clip(0, 1)

    bull = np.clip(
        0.7 * np.tanh(alignment * 5) + 0.2 * np.maximum(directional, 0) + 0.1 * trend_strength,
        0,
        1,
    ).astype("float32")
    bear = np.clip(
        0.7 * np.tanh(-alignment * 5) + 0.2 * np.maximum(-directional, 0) + 0.1 * trend_strength,
        0,
        1,
    ).astype("float32")
    sideways = (1.0 - np.clip(bull + bear, 0, 1)).astype("float32")

    return bull, bear, sideways


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    direction = df["Close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * df["Volume"]).cumsum()


def calculate_stochastic(
    df: pd.DataFrame, k_window: int = 14, d_window: int = 3
) -> tuple[pd.Series, pd.Series]:
    low_min = df["Low"].rolling(window=k_window, min_periods=1).min()
    high_max = df["High"].rolling(window=k_window, min_periods=1).max()
    denom = (high_max - low_min).replace(0, np.nan)
    pct_k = (100 * ((df["Close"] - low_min) / denom)).fillna(50)
    return pct_k, pct_k.rolling(window=d_window, min_periods=1).mean()


def calculate_williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_max = df["High"].rolling(window=window, min_periods=1).max()
    low_min = df["Low"].rolling(window=window, min_periods=1).min()
    denom = (high_max - low_min).replace(0, np.nan)
    return (-100 * ((high_max - df["Close"]) / denom)).fillna(-50)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 전부 추가."""
    df = df.copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    close = df["Close"]
    new_cols: dict[str, Any] = {}

    ma5 = calculate_sma(close, 5).astype("float32")
    ma10 = calculate_sma(close, 10).astype("float32")
    ma20 = calculate_sma(close, 20).astype("float32")
    ma50 = calculate_sma(close, 50).astype("float32")
    ma200 = calculate_sma(close, 200).astype("float32")
    ema12 = calculate_ema(close, 12).astype("float32")
    ema26 = calculate_ema(close, 26).astype("float32")
    new_cols.update(
        {
            "MA5": ma5,
            "MA10": ma10,
            "MA20": ma20,
            "MA50": ma50,
            "MA200": ma200,
            "EMA12": ema12,
            "EMA26": ema26,
        }
    )

    new_cols["RSI14"] = calculate_rsi(close, 14).astype("float32")
    new_cols["RSI7"] = calculate_rsi(close, 7).astype("float32")

    macd, macd_sig, macd_hist = calculate_macd(close)
    new_cols["MACD"] = macd.astype("float32")
    new_cols["MACD_Signal"] = macd_sig.astype("float32")
    new_cols["MACD_Hist"] = macd_hist.astype("float32")

    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close)
    bb_width = ((bb_upper - bb_lower) / bb_mid).astype("float32")
    denom = (bb_upper - bb_lower).replace(0, np.nan)
    bb_pos = ((close - bb_lower) / denom).fillna(0.5).clip(0, 1).astype("float32")
    new_cols.update(
        {
            "BB_Upper": bb_upper.astype("float32"),
            "BB_Middle": bb_mid.astype("float32"),
            "BB_Lower": bb_lower.astype("float32"),
            "BB_Width": bb_width,
            "BB_Position": bb_pos,
        }
    )

    atr14 = calculate_atr(df, 14).astype("float32")
    new_cols["ATR14"] = atr14
    new_cols["ATR_Pct"] = (atr14 / close * 100).astype("float32")

    stk, std_ = calculate_stochastic(df)
    new_cols["STOCH_K"] = stk.astype("float32")
    new_cols["STOCH_D"] = std_.astype("float32")

    new_cols["WILLIAMS_R"] = calculate_williams_r(df).astype("float32")

    obv = calculate_obv(df).astype("float32")
    obv_ema = calculate_ema(obv, 10).astype("float32")
    obv_trend = ((obv - obv_ema) / obv_ema.abs().replace(0, np.nan)).astype("float32")
    new_cols.update({"OBV": obv, "OBV_EMA": obv_ema, "OBV_Trend": obv_trend.fillna(0)})

    vol_sma20 = df["Volume"].rolling(20, min_periods=1).mean().astype("float32")
    vol_ratio = (df["Volume"] / vol_sma20.replace(0, np.nan)).fillna(1).astype("float32")
    new_cols.update({"Volume_SMA20": vol_sma20, "Volume_Ratio": vol_ratio})

    for n in [1, 3, 5, 10, 20]:
        new_cols[f"Return_{n}d"] = close.pct_change(n, fill_method=None).astype("float32")

    new_cols["Price_vs_MA20"] = ((close - ma20) / ma20).astype("float32")
    new_cols["Price_vs_MA50"] = ((close - ma50) / ma50).astype("float32")
    new_cols["Price_vs_MA200"] = ((close - ma200) / ma200).astype("float32")
    new_cols["MA5_vs_MA20"] = ((ma5 - ma20) / ma20).astype("float32")
    new_cols["MA20_vs_MA50"] = ((ma20 - ma50) / ma50).astype("float32")

    plus_di, minus_di, adx = calculate_directional_indicators(df)
    new_cols["Plus_DI"] = plus_di
    new_cols["Minus_DI"] = minus_di
    new_cols["ADX14"] = adx

    ma_alignment_spread = ((ma5 - ma20).abs() + (ma20 - ma50).abs()) / ma50.replace(0, np.nan)
    new_cols["MA_Alignment_Spread"] = ma_alignment_spread.fillna(0).astype("float32")
    new_cols["MA_Alignment_Ratio"] = (
        ((ma5 - ma20) / ma20).fillna(0) + ((ma20 - ma50) / ma50).fillna(0)
    ).astype("float32")

    bull_prob, bear_prob, sideways_prob = calculate_regime_probabilities(
        ma5, ma20, ma50, plus_di, minus_di, adx
    )
    new_cols["Regime_Prob_Bull"] = bull_prob
    new_cols["Regime_Prob_Bear"] = bear_prob
    new_cols["Regime_Prob_Sideways"] = sideways_prob

    high20 = df["High"].rolling(20, min_periods=1).max().astype("float32")
    low20 = df["Low"].rolling(20, min_periods=1).min().astype("float32")
    pp20 = (
        ((close - low20) / (high20 - low20).replace(0, np.nan)).fillna(0.5).astype("float32")
    )
    new_cols.update({"High_20d": high20, "Low_20d": low20, "Price_Position_20d": pp20})

    op = df["Open"].astype("float32")
    hi = df["High"].astype("float32")
    lo = df["Low"].astype("float32")
    body_size = ((close - op).abs() / op.replace(0, np.nan)).fillna(0).astype("float32")
    body_top = pd.concat([op, close], axis=1).max(axis=1).astype("float32")
    body_bot = pd.concat([op, close], axis=1).min(axis=1).astype("float32")
    upper_shad = ((hi - body_top) / op.replace(0, np.nan)).fillna(0).astype("float32")
    lower_shad = ((body_bot - lo) / op.replace(0, np.nan)).fillna(0).astype("float32")
    new_cols.update(
        {
            "Body_Size": body_size,
            "Upper_Shadow": upper_shad,
            "Lower_Shadow": lower_shad,
            "Is_Bullish": (close > op).astype("int8"),
        }
    )

    mom10 = (close - close.shift(10)).astype("float32")
    mom_n = (mom10 / close.shift(10).replace(0, np.nan)).fillna(0).astype("float32")
    new_cols.update({"Momentum_10d": mom10, "Momentum_Normalized": mom_n})

    macd_above = macd > macd_sig
    cross = pd.Series(0, index=df.index, dtype="int8")
    prev_macd_above = macd_above.shift(1, fill_value=False)
    cross[macd_above & ~prev_macd_above] = 1
    cross[~macd_above & prev_macd_above] = -1
    new_cols["MACD_Cross"] = cross

    new_cols["Target"] = (close.shift(-1) > close).astype("int8")
    new_cols["Target_Return"] = (close.shift(-1) / close - 1).astype("float32")

    df = df.assign(**new_cols)
    return df


def get_current_signals(df: pd.DataFrame) -> dict[str, Any]:
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest
    signals = {}

    rsi = float(latest.get("RSI14", 50))
    if rsi > 70:
        signals["RSI"] = ("SELL", f"RSI {rsi:.1f} - 과매수", "red")
    elif rsi < 30:
        signals["RSI"] = ("BUY", f"RSI {rsi:.1f} - 과매도", "green")
    else:
        signals["RSI"] = ("HOLD", f"RSI {rsi:.1f} - 중립", "gray")

    macd = float(latest.get("MACD", 0))
    macd_sig_val = float(latest.get("MACD_Signal", 0))
    p_macd = float(prev.get("MACD", 0))
    p_sig = float(prev.get("MACD_Signal", 0))
    if macd > macd_sig_val and p_macd <= p_sig:
        signals["MACD"] = ("BUY", "MACD 골든 크로스", "green")
    elif macd < macd_sig_val and p_macd >= p_sig:
        signals["MACD"] = ("SELL", "MACD 데드 크로스", "red")
    elif macd > macd_sig_val:
        signals["MACD"] = ("BUY", "MACD 매수 구간", "green")
    else:
        signals["MACD"] = ("SELL", "MACD 매도 구간", "red")

    bb_pos = float(latest.get("BB_Position", 0.5))
    if bb_pos > 0.95:
        signals["Bollinger"] = ("SELL", "볼린저 상단 돌파 - 과매수", "red")
    elif bb_pos < 0.05:
        signals["Bollinger"] = ("BUY", "볼린저 하단 이탈 - 과매도", "green")
    else:
        signals["Bollinger"] = ("HOLD", f"볼린저 밴드 내 {bb_pos:.0%}", "gray")

    ma5 = float(latest.get("MA5", 0))
    ma20 = float(latest.get("MA20", 0))
    ma50 = float(latest.get("MA50", 0))
    if ma5 > ma20 > ma50:
        signals["MA"] = ("BUY", "정배열 (MA5>MA20>MA50)", "green")
    elif ma5 < ma20 < ma50:
        signals["MA"] = ("SELL", "역배열 (MA5<MA20<MA50)", "red")
    else:
        signals["MA"] = ("HOLD", "이동평균 혼조세", "gray")

    stk = float(latest.get("STOCH_K", 50))
    if stk > 80:
        signals["Stochastic"] = ("SELL", f"Stoch %K {stk:.1f} - 과매수", "red")
    elif stk < 20:
        signals["Stochastic"] = ("BUY", f"Stoch %K {stk:.1f} - 과매도", "green")
    else:
        signals["Stochastic"] = ("HOLD", f"Stoch %K {stk:.1f} - 중립", "gray")

    vr = float(latest.get("Volume_Ratio", 1))
    if vr > 2.0:
        signals["Volume"] = ("WATCH", f"거래량 급증 ({vr:.1f}배)", "orange")
    elif vr > 1.5:
        signals["Volume"] = ("WATCH", f"거래량 증가 ({vr:.1f}배)", "yellow")
    else:
        signals["Volume"] = ("HOLD", f"거래량 보통 ({vr:.1f}배)", "gray")

    return signals


def get_support_resistance(df: pd.DataFrame, window: int = 20) -> dict[str, Any]:
    close = df["Close"]
    current = float(close.iloc[-1])
    recent = df.tail(window)
    yearly = df.tail(252)
    prev = df.iloc[-2]
    pivot = (float(prev["High"]) + float(prev["Low"]) + float(prev["Close"])) / 3
    return {
        "current": current,
        "resistance_20d": float(recent["High"].max()),
        "support_20d": float(recent["Low"].min()),
        "resistance_52w": float(yearly["High"].max()),
        "support_52w": float(yearly["Low"].min()),
        "pivot": pivot,
        "pivot_r1": 2 * pivot - float(prev["Low"]),
        "pivot_s1": 2 * pivot - float(prev["High"]),
    }
