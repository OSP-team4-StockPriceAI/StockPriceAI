"""
앙상블 예측 모듈
구조:
  XGBoostPredictor   — 피처 기반 베이스라인 (항상 실행)
  LSTMPredictor      — 시계열 딥러닝 (복잡 국면에서 추가)
  RegimeDetector     — 시장 국면 감지
  EnsemblePredictor  — XGB + LSTM 가중 결합

플랫폼 독립 — CPU 전용, EC2 컨테이너 환경 지원
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from ..core.config import (
    DATA,
    PYTORCH,
    PYTORCH_SCANNER,
    XGBOOST,
    XGBOOST_SCANNER,
)

warnings.filterwarnings("ignore")

log = logging.getLogger("stockai.ml")


def _bar(current: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "[" + "?" * width + "]"
    filled = int(width * current / total)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {current}/{total}"


# ─────────────────────────────────────────────────────────────
# 피처 정의
# ─────────────────────────────────────────────────────────────

BASE_FEATURES = [
    "RSI14", "RSI7",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Width", "BB_Position",
    "ATR_Pct",
    "STOCH_K", "STOCH_D",
    "WILLIAMS_R",
    "Volume_Ratio", "OBV_Trend",
    "Return_1d", "Return_3d", "Return_5d", "Return_10d", "Return_20d",
    "Price_vs_MA20", "Price_vs_MA50",
    "MA5_vs_MA20", "MA20_vs_MA50",
    "MA_Alignment_Spread", "MA_Alignment_Ratio",
    "ADX14", "Plus_DI", "Minus_DI",
    "Regime_Prob_Bull", "Regime_Prob_Bear", "Regime_Prob_Sideways",
    "Price_Position_20d",
    "Body_Size", "Upper_Shadow", "Lower_Shadow", "Is_Bullish",
    "Momentum_Normalized", "MACD_Cross",
]
SENTIMENT_FEATURES = ["Sentiment_Score", "Sentiment_Positive", "Sentiment_Negative"]


def get_feature_columns(df: pd.DataFrame, include_sentiment: bool = True) -> list[str]:
    cols = [f for f in BASE_FEATURES if f in df.columns]
    if include_sentiment:
        cols += [f for f in SENTIMENT_FEATURES if f in df.columns]
    return cols


def _auto_params(n_samples: int) -> dict[str, Any]:
    max_train = DATA["max_train_samples"]
    max_lstm = DATA["max_lstm_samples"]

    if n_samples < 300:
        return {
            "n_estimators_xgb": 150, "n_splits": 3, "lstm_epochs": 60, "max_lstm_samples": max_lstm
        }
    elif n_samples < 800:
        return {
            "n_estimators_xgb": 200, "n_splits": 4, "lstm_epochs": 80, "max_lstm_samples": max_lstm
        }
    elif n_samples < 2000:
        return {
            "n_estimators_xgb": 300, "n_splits": 5, "lstm_epochs": 100, "max_lstm_samples": max_lstm
        }
    elif n_samples <= max_train:
        return {
            "n_estimators_xgb": 300, "n_splits": 5, "lstm_epochs": 120, "max_lstm_samples": max_lstm
        }
    else:
        return {
            "n_estimators_xgb": 300,
            "n_splits": 5,
            "lstm_epochs": 100,
            "fast": True,
            "max_samples": max_train,
            "max_lstm_samples": max_lstm,
        }


def prepare_training_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_samples: int = 60,
    max_samples: int | None = None,
) -> tuple[npt.NDArray[Any] | None, npt.NDArray[Any] | None, pd.DatetimeIndex | None]:
    work = df[feature_cols + ["Target"]].copy()
    work = work.dropna(subset=["Target"]).iloc[:-1]

    if len(work) < min_samples:
        return None, None, None

    if max_samples and len(work) > max_samples:
        work = work.iloc[-max_samples:]

    X = work[feature_cols].ffill().bfill().fillna(0)
    X = X.replace([np.inf, -np.inf], 0).clip(-10, 10)
    return X.values.astype(np.float32), work["Target"].values.astype(int), work.index


# ─────────────────────────────────────────────────────────────
# RegimeDetector
# ─────────────────────────────────────────────────────────────

class RegimeDetector:
    """시장 국면 감지 (단순 / 복잡)"""

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def compute(self, df: pd.DataFrame) -> dict[str, Any]:
        recent = df.tail(self.lookback).copy()
        scores: dict[str, float] = {}

        if "ATR_Pct" in recent.columns:
            atr_series = recent["ATR_Pct"].dropna()
            if len(atr_series) > 10:
                atr_cv = atr_series.std() / (atr_series.mean() + 1e-9)
                atr_spike = atr_series.tail(10).mean() / (atr_series.mean() + 1e-9)
                scores["volatility"] = float(np.clip(atr_cv * 0.5 + (atr_spike - 1) * 0.3, 0, 1))
            else:
                scores["volatility"] = 0.3
        else:
            scores["volatility"] = 0.3

        if all(c in recent.columns for c in ["MA5_vs_MA20", "MA20_vs_MA50"]):
            ma_fast = recent["MA5_vs_MA20"].dropna()
            ma_slow = recent["MA20_vs_MA50"].dropna()
            if len(ma_fast) > 5:
                fast_flips = int((np.diff(np.sign(ma_fast.values)) != 0).sum())
                slow_flips = int((np.diff(np.sign(ma_slow.values)) != 0).sum())
                flip_rate = (fast_flips + slow_flips) / (len(ma_fast) * 2 + 1e-9)
                scores["trend_inconsistency"] = float(np.clip(flip_rate * 3, 0, 1))
            else:
                scores["trend_inconsistency"] = 0.3
        else:
            scores["trend_inconsistency"] = 0.3

        if "RSI14" in recent.columns:
            rsi = recent["RSI14"].dropna()
            if len(rsi) > 5:
                scores["rsi_extremes"] = float(
                    np.clip(((rsi > 70) | (rsi < 30)).mean() * 2.5, 0, 1)
                )
            else:
                scores["rsi_extremes"] = 0.2
        else:
            scores["rsi_extremes"] = 0.2

        if "MACD_Cross" in recent.columns:
            cross_count = recent["MACD_Cross"].abs().sum()
            scores["macd_cross_freq"] = float(
                np.clip(cross_count / (self.lookback + 1e-9) * 15, 0, 1)
            )
        else:
            scores["macd_cross_freq"] = 0.2

        if "Momentum_Normalized" in recent.columns:
            mom = recent["Momentum_Normalized"].dropna()
            if len(mom) > 5:
                mom_flips = int((np.diff(np.sign(mom.values)) != 0).sum())
                scores["momentum_reversal"] = float(np.clip(mom_flips / len(mom) * 4, 0, 1))
            else:
                scores["momentum_reversal"] = 0.2
        else:
            scores["momentum_reversal"] = 0.2

        if "BB_Position" in recent.columns:
            bb = recent["BB_Position"].dropna()
            if len(bb) > 5:
                scores["bb_breakout"] = float(np.clip(((bb > 0.95) | (bb < 0.05)).mean() * 5, 0, 1))
            else:
                scores["bb_breakout"] = 0.2
        else:
            scores["bb_breakout"] = 0.2

        if "ADX14" in recent.columns:
            adx = recent["ADX14"].dropna()
            if len(adx) > 5:
                scores["trend_strength"] = float(np.clip(adx.mean() / 50, 0, 1))
            else:
                scores["trend_strength"] = 0.3
        else:
            scores["trend_strength"] = 0.3

        if "MA_Alignment_Spread" in recent.columns:
            spread = recent["MA_Alignment_Spread"].dropna()
            if len(spread) > 5:
                scores["alignment_strength"] = float(np.clip((spread.mean() - 0.02) / 0.15, 0, 1))
            else:
                scores["alignment_strength"] = 0.3
        else:
            scores["alignment_strength"] = 0.3

        weights = {
            "volatility": 0.25,
            "trend_inconsistency": 0.20,
            "trend_strength": 0.15,
            "alignment_strength": 0.15,
            "rsi_extremes": 0.10,
            "macd_cross_freq": 0.10,
            "momentum_reversal": 0.03,
            "bb_breakout": 0.02,
        }
        complexity = float(np.clip(sum(scores.get(k, 0) * w for k, w in weights.items()), 0, 1))

        if complexity < 0.30:
            regime = "simple"
        elif complexity < 0.60:
            regime = "moderate"
        else:
            regime = "complex"

        return {
            "complexity": round(complexity, 3),
            "regime": regime,
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "use_lstm": regime in ("moderate", "complex"),
        }


# ─────────────────────────────────────────────────────────────
# XGBoostPredictor
# ─────────────────────────────────────────────────────────────

class XGBoostPredictor:
    def __init__(self, scanner_mode: bool = False):
        self.scanner_mode = scanner_mode
        self.model: Any = None
        self.scaler = StandardScaler()
        self.feature_cols: list[str] | None = None
        self.is_trained = False
        self.feature_importances_: pd.Series | None = None
        self.training_metrics: dict[str, Any] = {}
        self._cv_proba: npt.NDArray[Any] | None = None

    def train(
        self, df: pd.DataFrame, include_sentiment: bool = True, n_splits: int = 5
    ) -> dict[str, Any]:
        try:
            import xgboost as xgb

            _ = xgb.XGBClassifier
        except Exception:
            return self._train_sklearn(df, include_sentiment)

        xgb_cfg = XGBOOST_SCANNER if self.scanner_mode else XGBOOST
        self.feature_cols = get_feature_columns(df, include_sentiment)

        raw_len = len(df)
        ap = _auto_params(raw_len)
        max_samp = ap.get("max_samples")
        n_est_cv = max(80, ap["n_estimators_xgb"] - 100) if not self.scanner_mode else 100
        n_est_fin = ap["n_estimators_xgb"] if not self.scanner_mode else 150
        n_splits_ = ap["n_splits"] if not self.scanner_mode else 3

        X, y, _ = prepare_training_data(df, self.feature_cols, max_samples=max_samp)
        if X is None or y is None:
            return {"error": "학습 데이터 부족 (최소 60일 필요)"}

        log.info(
            f"XGBoost 학습: 데이터={raw_len}일, 피처={len(self.feature_cols)}개, CV={n_splits_}fold"
        )

        t0 = time.time()
        tscv = TimeSeriesSplit(n_splits=n_splits_)
        cv_scores = []
        oof_proba = np.full(len(y), 0.5)

        for fold_i, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr_idx])
            Xvl = sc.transform(X[val_idx])

            m = xgb.XGBClassifier(
                n_estimators=n_est_cv,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                eval_metric="logloss",
                random_state=42,
                n_jobs=xgb_cfg["nthread"],
                verbosity=0,
                device=xgb_cfg["device"],
                tree_method=xgb_cfg["tree_method"],
                max_bin=xgb_cfg["max_bin"],
                grow_policy=xgb_cfg["grow_policy"],
            )
            m.fit(Xtr, y[tr_idx], eval_set=[(Xvl, y[val_idx])], verbose=False)
            oof_proba[val_idx] = m.predict_proba(Xvl)[:, 1]
            fold_acc = accuracy_score(y[val_idx], m.predict(Xvl))
            cv_scores.append(fold_acc)

        self._cv_proba = oof_proba
        log.info(f"XGBoost CV 평균 정확도: {float(np.mean(cv_scores)):.3f} ({time.time()-t0:.1f}s)")

        X_sc = self.scaler.fit_transform(X)
        self.model = xgb.XGBClassifier(
            n_estimators=n_est_fin,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=xgb_cfg["nthread"],
            verbosity=0,
            device=xgb_cfg["device"],
            tree_method=xgb_cfg["tree_method"],
            max_bin=xgb_cfg["max_bin"],
            grow_policy=xgb_cfg["grow_policy"],
        )
        self.model.fit(X_sc, y, verbose=False)
        self.is_trained = True

        self.feature_importances_ = pd.Series(
            self.model.feature_importances_, index=self.feature_cols
        ).sort_values(ascending=False)

        y_pred = self.model.predict(X_sc)
        self.training_metrics = {
            "cv_accuracy_mean": float(np.mean(cv_scores)),
            "cv_accuracy_std": float(np.std(cv_scores)),
            "train_accuracy": float(accuracy_score(y, y_pred)),
            "model_type": "XGBoost",
            "n_features": len(self.feature_cols),
            "n_samples": len(y),
            "n_samples_total": raw_len,
        }
        return self.training_metrics

    def _train_sklearn(self, df: pd.DataFrame, include_sentiment: bool) -> dict[str, Any]:
        from sklearn.ensemble import GradientBoostingClassifier

        log.warning("XGBoost 로드 실패 → GradientBoosting 폴백")
        self.feature_cols = get_feature_columns(df, include_sentiment)
        X, y, _ = prepare_training_data(df, self.feature_cols)
        if X is None or y is None:
            return {"error": "학습 데이터 부족"}

        X_sc = self.scaler.fit_transform(X)
        self.model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42
        )
        self.model.fit(X_sc, y)
        self.is_trained = True

        self.feature_importances_ = pd.Series(
            self.model.feature_importances_, index=self.feature_cols
        ).sort_values(ascending=False)

        y_pred = self.model.predict(X_sc)
        self.training_metrics = {
            "train_accuracy": float(accuracy_score(y, y_pred)),
            "model_type": "GradientBoosting (sklearn)",
            "n_features": len(self.feature_cols),
            "n_samples": len(y),
        }
        return self.training_metrics

    def predict_proba(self, df: pd.DataFrame) -> float | None:
        if not self.is_trained or self.feature_cols is None:
            return None
        try:
            latest = df[self.feature_cols].iloc[-1:]
            latest = latest.ffill().bfill().fillna(0).replace([np.inf, -np.inf], 0)
            X_sc = self.scaler.transform(latest.values)
            return float(self.model.predict_proba(X_sc)[0, 1])
        except Exception:
            return None

    def predict(self, df: pd.DataFrame) -> dict[str, Any]:
        p = self.predict_proba(df)
        if p is None:
            return {"error": "예측 실패"}
        return _build_result(p, self.training_metrics.get("model_type", "XGBoost"))


# ─────────────────────────────────────────────────────────────
# LSTMPredictor — CPU 전용 (EC2 환경)
# ─────────────────────────────────────────────────────────────

class LSTMPredictor:
    """LSTM 시계열 예측기. CPU 전용 (MPS/CUDA 없는 EC2 환경)."""

    def __init__(self, sequence_length: int = 20, scanner_mode: bool = False):
        self.sequence_length = sequence_length
        self.scanner_mode = scanner_mode
        self.model: Any = None
        self.scaler = StandardScaler()
        self.feature_cols: list[str] | None = None
        self.is_trained = False
        self.framework: str | None = None
        self.training_metrics: dict[str, Any] = {}

    @staticmethod
    def available_framework() -> str | None:
        try:
            import torch as _t  # noqa: F401

            return "pytorch"
        except Exception:
            pass
        try:
            import tensorflow as _tf  # noqa: F401

            return "tensorflow"
        except Exception:
            pass
        return None

    def train(self, df: pd.DataFrame, include_sentiment: bool = True) -> dict[str, Any]:
        fw = self.available_framework()
        if not fw:
            return {"error": "PyTorch / TensorFlow 미설치"}

        self.framework = fw
        self.feature_cols = get_feature_columns(df, include_sentiment)
        X, y, _ = prepare_training_data(df, self.feature_cols)

        if X is None or y is None or len(X) < self.sequence_length + 20:
            return {"error": f"LSTM 학습 데이터 부족 (최소 {self.sequence_length + 20}일 필요)"}

        X_sc = self.scaler.fit_transform(X)

        if fw == "pytorch":
            return self._train_pytorch(X_sc, y)
        return self._train_tensorflow(X_sc, y)

    def _train_pytorch(self, X_sc: npt.NDArray[Any], y: npt.NDArray[Any]) -> dict[str, Any]:
        import torch
        import torch.nn as nn

        # CPU 전용 — scanner_mode 여부와 무관하게 CPU 사용
        pt_cfg = PYTORCH_SCANNER if self.scanner_mode else PYTORCH
        device = torch.device(str(pt_cfg["device"]))
        torch.set_num_threads(cast(int, pt_cfg["num_threads"]))

        SEQ = self.sequence_length
        X_seq = np.array([X_sc[i - SEQ:i] for i in range(SEQ, len(X_sc))])
        y_seq = y[SEQ:]
        split = int(len(X_seq) * 0.8)

        def to_t(a: npt.NDArray[Any]) -> Any:
            return torch.tensor(a, dtype=torch.float32, device=device)

        Xtr, Xvl = to_t(X_seq[:split]), to_t(X_seq[split:])
        ytr, yvl = to_t(y_seq[:split].astype("float32")), to_t(y_seq[split:].astype("float32"))
        batch_sz = cast(int, pt_cfg["batch_size"])
        n_feat = X_seq.shape[2]

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm1 = nn.LSTM(n_feat, 64, batch_first=True, dropout=0.2, num_layers=1)
                self.ln1 = nn.LayerNorm(64)
                self.lstm2 = nn.LSTM(64, 32, batch_first=True)
                self.drop = nn.Dropout(0.2)
                self.fc = nn.Sequential(
                    nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
                )

            def forward(self, x: Any) -> Any:
                o, _ = self.lstm1(x)
                o = self.ln1(o[:, -1, :]).unsqueeze(1)
                o, _ = self.lstm2(o)
                return self.fc(self.drop(o[:, -1, :])).squeeze(1)

        net = _Net().to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
        crit = nn.BCELoss()
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

        max_epochs = _auto_params(len(X_seq))["lstm_epochs"]
        patience = max(10, max_epochs // 10)

        log.info(
            "LSTM (PyTorch/CPU) 학습: SEQ=%d, 피처=%d, 샘플=%d, 최대에포크=%d",
            SEQ, n_feat, len(X_seq), max_epochs,
        )

        t0 = time.time()
        best_val, best_state, wait = 0.0, None, 0

        for epoch in range(max_epochs):
            net.train()
            perm = torch.randperm(len(Xtr), device=device)
            for i in range(0, len(Xtr), batch_sz):
                idx = perm[i:i + batch_sz]
                opt.zero_grad()
                loss_b = crit(net(Xtr[idx]), ytr[idx])
                loss_b.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sched.step()

            net.eval()
            with torch.no_grad():
                val_acc = ((net(Xvl) > 0.5).float() == yvl).float().mean().item()
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state:
            net.load_state_dict(best_state)
        self.model = net
        self.is_trained = True
        log.info(f"LSTM 완료: best_val_acc={best_val:.3f} ({time.time()-t0:.1f}s)")

        self.training_metrics = {
            "val_accuracy": round(best_val, 4),
            "model_type": "LSTM (PyTorch/CPU)",
            "sequence_length": SEQ,
            "n_features": n_feat,
            "n_samples": len(y_seq),
        }
        return self.training_metrics

    def _train_tensorflow(self, X_sc: npt.NDArray[Any], y: npt.NDArray[Any]) -> dict[str, Any]:

        try:
            from tensorflow import keras
        except Exception:
            import tensorflow.keras as keras

        SEQ = self.sequence_length
        X_seq = np.array([X_sc[i - SEQ:i] for i in range(SEQ, len(X_sc))])
        y_seq = y[SEQ:]
        split = int(len(X_seq) * 0.8)

        inp = keras.Input(shape=(SEQ, X_seq.shape[2]))
        x = keras.layers.LSTM(64, return_sequences=True)(inp)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.LSTM(32)(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(16, activation="relu")(x)
        out = keras.layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"]
        )

        n_ep = _auto_params(len(X_seq))["lstm_epochs"]
        cb_es = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=max(10, n_ep // 10), restore_best_weights=True
        )
        hist = model.fit(
            X_seq[:split],
            y_seq[:split],
            validation_data=(X_seq[split:], y_seq[split:]),
            epochs=n_ep,
            batch_size=32,
            callbacks=[cb_es],
            verbose=0,
        )
        self.model = model
        self.is_trained = True
        val_acc = max(hist.history.get("val_accuracy", [0.5]))

        self.training_metrics = {
            "val_accuracy": float(val_acc),
            "model_type": "LSTM (TensorFlow/CPU)",
            "sequence_length": SEQ,
            "n_features": X_seq.shape[2],
            "n_samples": len(y_seq),
        }
        return self.training_metrics

    def predict_proba(self, df: pd.DataFrame) -> float | None:
        if not self.is_trained or self.feature_cols is None:
            return None
        try:
            SEQ = self.sequence_length
            recent = df[self.feature_cols].tail(SEQ).copy()
            recent = recent.ffill().bfill().fillna(0).replace([np.inf, -np.inf], 0)
            if len(recent) < SEQ:
                return None

            X_sc = self.scaler.transform(recent.values)
            X_seq = X_sc.reshape(1, SEQ, -1)

            if self.framework == "pytorch":
                import torch

                device = next(self.model.parameters()).device
                self.model.eval()
                with torch.no_grad():
                    t = torch.tensor(X_seq, dtype=torch.float32, device=device)
                    return float(self.model(t)[0].cpu().item())
            else:
                return float(self.model.predict(X_seq, verbose=0)[0, 0])
        except Exception:
            return None

    def predict(self, df: pd.DataFrame) -> dict[str, Any]:
        p = self.predict_proba(df)
        if p is None:
            return {"error": "LSTM 예측 실패"}
        return _build_result(p, self.training_metrics.get("model_type", "LSTM"))


# ─────────────────────────────────────────────────────────────
# EnsemblePredictor
# ─────────────────────────────────────────────────────────────

class EnsemblePredictor:
    """XGBoost + LSTM 동적 앙상블 예측기."""

    def __init__(
        self,
        sequence_length: int = 20,
        complexity_threshold: float = 0.30,
        min_lstm_weight: float = 0.20,
        max_lstm_weight: float = 0.55,
        scanner_mode: bool = False,
    ):
        self.sequence_length = sequence_length
        self.complexity_threshold = complexity_threshold
        self.min_lstm_weight = min_lstm_weight
        self.max_lstm_weight = max_lstm_weight
        self.scanner_mode = scanner_mode

        self.xgb = XGBoostPredictor(scanner_mode=scanner_mode)
        self.lstm = LSTMPredictor(sequence_length=sequence_length, scanner_mode=scanner_mode)
        self.regime = RegimeDetector()

        self.is_trained = False
        self.lstm_trained = False
        self.training_metrics: dict[str, Any] = {}
        self.feature_importances_: pd.Series | None = None

    def _compute_weights(self, complexity: float) -> tuple[float, float]:
        if not self.lstm_trained:
            return 1.0, 0.0

        raw_lstm_w = np.interp(
            complexity,
            [self.complexity_threshold, 1.0],
            [self.min_lstm_weight, self.max_lstm_weight],
        )

        xgb_acc = self.xgb.training_metrics.get(
            "cv_accuracy_mean", self.xgb.training_metrics.get("train_accuracy", 0.5)
        )
        lstm_acc = self.lstm.training_metrics.get("val_accuracy", 0.5)
        acc_ratio = float(np.clip(lstm_acc / (xgb_acc + 1e-9), 0.7, 1.3))

        lstm_w = float(np.clip(raw_lstm_w * acc_ratio, self.min_lstm_weight, self.max_lstm_weight))
        return round(1.0 - lstm_w, 3), round(lstm_w, 3)

    def train(
        self,
        df: pd.DataFrame,
        include_sentiment: bool = True,
        force_lstm: bool = False,
    ) -> dict[str, Any]:
        ap_ens = _auto_params(len(df))
        t_total = time.time()

        # Step 1: XGBoost
        xgb_metrics = self.xgb.train(df, include_sentiment=include_sentiment)
        if "error" in xgb_metrics:
            return xgb_metrics
        self.is_trained = True
        self.feature_importances_ = self.xgb.feature_importances_

        # Step 2: 국면 감지
        regime_info = self.regime.compute(df)
        complexity = regime_info["complexity"]
        use_lstm = force_lstm or regime_info["use_lstm"]

        # Step 3: LSTM (조건부)
        lstm_metrics: dict[str, Any] = {}
        lstm_fw = LSTMPredictor.available_framework()
        max_lstm = ap_ens.get("max_lstm_samples", DATA["max_lstm_samples"])
        lstm_df = df.iloc[-max_lstm:] if len(df) > max_lstm else df

        if use_lstm and lstm_fw:
            lstm_metrics = self.lstm.train(lstm_df, include_sentiment=include_sentiment)
            if "error" not in lstm_metrics:
                self.lstm_trained = True
                log.info(f"LSTM 완료: val_acc={lstm_metrics.get('val_accuracy', 0):.3f}")
            else:
                self.lstm_trained = False
                log.warning(f"LSTM 실패: {lstm_metrics.get('error')} → XGBoost 단독")
        else:
            self.lstm_trained = False
            log.info(f"LSTM 스킵 (국면={regime_info['regime']}, fw={lstm_fw})")

        # Step 4: 가중치
        w_xgb, w_lstm = self._compute_weights(complexity)

        self.training_metrics = {
            "model_type": self._model_label(w_xgb, w_lstm),
            "xgb_metrics": xgb_metrics,
            "lstm_metrics": lstm_metrics,
            "regime": regime_info,
            "w_xgb": w_xgb,
            "w_lstm": w_lstm,
            "lstm_active": self.lstm_trained,
            "lstm_framework": lstm_fw or "없음",
            "n_samples": xgb_metrics.get("n_samples", 0),
            "n_features": xgb_metrics.get("n_features", 0),
            "cv_accuracy_mean": xgb_metrics.get(
                "cv_accuracy_mean", xgb_metrics.get("train_accuracy", 0)
            ),
            "elapsed_sec": round(time.time() - t_total, 1),
        }
        if self.lstm_trained:
            self.training_metrics["lstm_val_accuracy"] = lstm_metrics.get("val_accuracy", 0)

        return self.training_metrics

    def _model_label(self, w_xgb: float, w_lstm: float) -> str:
        if w_lstm == 0.0:
            return "XGBoost (단독)"
        return f"Ensemble (XGB {int(round(w_xgb*100))}% + LSTM {int(round(w_lstm*100))}%)"

    def predict(self, df: pd.DataFrame) -> dict[str, Any]:
        if not self.is_trained:
            return {"error": "모델 미학습"}

        p_xgb = self.xgb.predict_proba(df)
        if p_xgb is None:
            return {"error": "XGBoost 예측 실패"}

        p_lstm: float | None = None
        if self.lstm_trained:
            p_lstm = self.lstm.predict_proba(df)

        regime_now = self.regime.compute(df)
        complexity = regime_now["complexity"]
        w_xgb, w_lstm = self._compute_weights(complexity)

        if p_lstm is None:
            w_xgb, w_lstm = 1.0, 0.0
            p_final = p_xgb
        else:
            p_final = w_xgb * p_xgb + w_lstm * p_lstm

        if p_lstm is not None and (p_xgb > 0.5) != (p_lstm > 0.5):
            p_final = 0.5 + (p_final - 0.5) * 0.6

        result = _build_result(p_final, self.training_metrics.get("model_type", "Ensemble"))
        result["ensemble_detail"] = {
            "p_xgb": round(p_xgb, 4),
            "p_lstm": round(p_lstm, 4) if p_lstm is not None else None,
            "w_xgb": w_xgb,
            "w_lstm": w_lstm,
            "complexity": complexity,
            "regime": regime_now["regime"],
        }
        return result


# ─────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────

def _build_result(up_prob: float, model_name: str) -> dict[str, Any]:
    down_prob = 1.0 - up_prob
    confidence = max(up_prob, down_prob)
    if up_prob > 0.65:
        signal = "BUY"
    elif down_prob > 0.55:
        signal = "SELL"
    else:
        signal = "HOLD"
    return {
        "direction": 1 if up_prob >= 0.5 else 0,
        "up_probability": round(up_prob, 4),
        "down_probability": round(down_prob, 4),
        "confidence": round(confidence, 4),
        "signal": signal,
        "model": model_name,
    }


def run_backtest(
    df: pd.DataFrame,
    predictor: Any,
    initial_capital: float = 10000,
    commission_rate: float = 0.001,
) -> dict[str, Any]:
    """Walk-forward 백테스트."""
    if not predictor.is_trained:
        return {}

    close = df["Close"]
    capital = initial_capital
    shares = 0
    position = "NONE"
    trades = []
    portfolio_values = []
    min_train = 60

    for i in range(min_train, len(df)):
        current_df = df.iloc[: i + 1]
        pred = predictor.predict(current_df)

        if "error" in pred:
            portfolio_values.append(
                capital + (shares * float(close.iloc[i]) if position == "LONG" else 0)
            )
            continue

        sig = pred["signal"]
        price = float(close.iloc[i])

        if sig == "BUY" and position == "NONE":
            shares = int(capital * (1 - commission_rate) / price)
            if shares > 0:
                capital -= shares * price * (1 + commission_rate)
                position = "LONG"
                trades.append({"type": "BUY", "price": price, "shares": shares})

        elif sig == "SELL" and position == "LONG":
            capital += shares * price * (1 - commission_rate)
            position = "NONE"
            trades.append({"type": "SELL", "price": price, "shares": shares})
            shares = 0

        portfolio_values.append(capital + (shares * price if position == "LONG" else 0))

    if position == "LONG" and shares > 0:
        capital += shares * float(close.iloc[-1]) * (1 - commission_rate)

    return {
        "initial_capital": initial_capital,
        "final_capital": round(capital, 2),
        "strategy_return_pct": round((capital / initial_capital - 1) * 100, 2),
        "n_trades": len(trades),
        "portfolio_values": portfolio_values,
    }
