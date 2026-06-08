"""
LSTM-XGBoost Stacking 예측 모듈
구조:
  LSTMFirstStackingPredictor  — LSTM을 먼저 학습하고, 그 예측 확률을 피처로 추가하여 XGBoost를 학습시키는 스태킹 모델
  NewXGBoostPredictor         — 기존 XGBoostPredictor를 상속받아 커스텀 피처 목록(LSTM_Proba 포함)을 받아 학습하도록 개선
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
from .predictor import (
    XGBoostPredictor,
    LSTMPredictor,
    prepare_training_data,
    get_feature_columns,
    _build_result,
    _auto_params,
)

warnings.filterwarnings("ignore")
log = logging.getLogger("stockai.ml")


def predict_lstm_history_proba(lstm_pred: LSTMPredictor, df: pd.DataFrame) -> pd.Series:
    """학습된 LSTMPredictor를 사용하여 입력 데이터 전체에 대한 LSTM 예측 확률을 생성합니다."""
    if not lstm_pred.is_trained or lstm_pred.feature_cols is None:
        return pd.Series(0.5, index=df.index)

    SEQ = lstm_pred.sequence_length
    feature_cols = lstm_pred.feature_cols

    work = df[feature_cols].copy()
    work = work.ffill().bfill().fillna(0)
    work = work.replace([np.inf, -np.inf], 0).clip(-10, 10)
    if len(work) < SEQ:
        return pd.Series(0.5, index=df.index)

    X = work.values.astype(np.float32)
    X_sc = lstm_pred.scaler.transform(X)
    
    probs = _predict_lstm_on_scaled_features(lstm_pred, X_sc, SEQ)
    return pd.Series(probs, index=df.index)


def _split_time_series_train_val(
    X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """시계열 데이터에서 마지막 val_ratio 만큼을 검증 데이터로 분리합니다."""
    val_count = max(1, int(len(X) * val_ratio))
    split = len(X) - val_count
    return X[:split], y[:split], X[split:], y[split:]


def _split_time_series_df(
    df: pd.DataFrame, val_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """시계열 데이터프레임을 마지막 val_ratio 만큼의 검증 데이터로 나눕니다."""
    val_count = max(1, int(len(df) * val_ratio))
    split = len(df) - val_count
    return df.iloc[:split], df.iloc[split:]


def _predict_lstm_on_scaled_features(
    lstm_pred: LSTMPredictor,
    X_sc: npt.NDArray[Any],
    sequence_length: int,
) -> npt.NDArray[Any]:
    """이미 스케일링된 배열 X_sc에 대해서만 LSTM 추론을 수행합니다 (초고속)."""
    probs = np.full(len(X_sc), 0.5, dtype=float)
    
    if len(X_sc) < sequence_length:
        return probs
    
    X_seq = np.array([X_sc[i - sequence_length:i] for i in range(sequence_length, len(X_sc))])
    
    if lstm_pred.framework == "pytorch":
        import torch
        device = next(lstm_pred.model.parameters()).device
        lstm_pred.model.eval()
        with torch.no_grad():
            t = torch.tensor(X_seq, dtype=torch.float32, device=device)
            preds = lstm_pred.model(t).cpu().numpy().flatten()
            probs[sequence_length:] = preds
    else:
        preds = lstm_pred.model.predict(X_seq, verbose=0).flatten()
        probs[sequence_length:] = preds
    
    return probs


def _compute_oof_lstm_proba(
    df: pd.DataFrame,
    sequence_length: int,
    include_sentiment: bool,
    n_splits: int = 5,
    scanner_mode: bool = False,
) -> pd.Series:
    """시간 순서대로 fold를 돌며 깨끗한 OOF LSTM 확률을 생성합니다 (최적화됨)."""
    oof = pd.Series(0.5, index=df.index, dtype=float)
    if len(df) < sequence_length + 20:
        return oof

    n_splits = min(n_splits, max(2, len(df) // (sequence_length + 1)))
    if n_splits < 2:
        return oof

    feature_cols = get_feature_columns(df, include_sentiment)
    
    # 루프 밖에서 한 번만 전체 df 준비 (스케일링, 결측치 처리)
    work = df[feature_cols].copy()
    work = work.ffill().bfill().fillna(0)
    work = work.replace([np.inf, -np.inf], 0).clip(-10, 10)
    X_full = work.values.astype(np.float32)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, val_idx in tscv.split(df):
        df_tr = df.iloc[train_idx]
        fold_lstm = LSTMPredictor(sequence_length=sequence_length, scanner_mode=scanner_mode)
        fold_metrics = fold_lstm.train(df_tr, include_sentiment=include_sentiment)
        if "error" in fold_metrics:
            continue

        # 준비된 X_full을 해당 폴드의 scaler로 변환하고, 필요한 부분만 추론
        start_idx = max(0, val_idx[0] - sequence_length)
        end_idx = val_idx[-1] + 1
        
        X_sc = fold_lstm.scaler.transform(X_full[start_idx:end_idx])
        fold_probs = _predict_lstm_on_scaled_features(fold_lstm, X_sc, sequence_length)
        
        # val_idx에 해당하는 인덱스만 OOF에 저장
        val_start_offset = val_idx[0] - start_idx
        val_end_offset = val_start_offset + len(val_idx)
        val_index = df.iloc[val_idx].index
        oof.loc[val_index] = fold_probs[val_start_offset:val_end_offset]

    return oof


class NewXGBoostPredictor(XGBoostPredictor):
    """지정된 커스텀 피처 리스트(예: LSTM_Proba 포함)로 학습할 수 있도록 확장한 XGBoost 예측기"""

    def train(
        self,
        df: pd.DataFrame,
        include_sentiment: bool = True,
        n_splits: int = 5,
        feature_cols: list[str] | None = None,
        cv_only: bool = False,
    ) -> dict[str, Any]:
        try:
            import xgboost as xgb

            _ = xgb.XGBClassifier
        except Exception:
            return self._train_sklearn(df, include_sentiment, feature_cols=feature_cols)

        xgb_cfg = XGBOOST_SCANNER if self.scanner_mode else XGBOOST
        
        if feature_cols is not None:
            self.feature_cols = feature_cols
        else:
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
            f"New XGBoost 학습: 데이터={raw_len}일, 피처={len(self.feature_cols)}개, CV={n_splits_}fold"
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
        log.info(f"New XGBoost CV 평균 정확도: {float(np.mean(cv_scores)):.3f} ({time.time()-t0:.1f}s)")

        if not cv_only:
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
            train_accuracy = float(accuracy_score(y, y_pred))
        else:
            train_accuracy = 0.0

        self.training_metrics = {
            "cv_accuracy_mean": float(np.mean(cv_scores)),
            "cv_accuracy_std": float(np.std(cv_scores)),
            "train_accuracy": train_accuracy,
            "model_type": "NewXGBoost",
            "n_features": len(self.feature_cols),
            "n_samples": len(y),
            "n_samples_total": raw_len,
        }
        return self.training_metrics

    def fit_full_data(
        self,
        df: pd.DataFrame,
        include_sentiment: bool = True,
        feature_cols: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            import xgboost as xgb

            _ = xgb.XGBClassifier
        except Exception:
            return self._train_sklearn(df, include_sentiment, feature_cols=feature_cols)

        xgb_cfg = XGBOOST_SCANNER if self.scanner_mode else XGBOOST

        if feature_cols is not None:
            self.feature_cols = feature_cols
        else:
            self.feature_cols = get_feature_columns(df, include_sentiment)

        raw_len = len(df)
        ap = _auto_params(raw_len)
        max_samp = ap.get("max_samples")
        n_est_fin = ap["n_estimators_xgb"] if not self.scanner_mode else 150

        X, y, _ = prepare_training_data(df, self.feature_cols, max_samples=max_samp)
        if X is None or y is None:
            return {"error": "학습 데이터 부족 (최소 60일 필요)"}

        self.scaler = StandardScaler()
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
            "train_accuracy": float(accuracy_score(y, y_pred)),
            "model_type": "NewXGBoost (full)",
            "n_features": len(self.feature_cols),
            "n_samples": len(y),
            "n_samples_total": raw_len,
        }
        return self.training_metrics

    def _train_sklearn(
        self,
        df: pd.DataFrame,
        include_sentiment: bool,
        feature_cols: list[str] | None = None,
    ) -> dict[str, Any]:
        from sklearn.ensemble import GradientBoostingClassifier

        log.warning("XGBoost 로드 실패 → GradientBoosting 폴백")
        if feature_cols is not None:
            self.feature_cols = feature_cols
        else:
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


class LSTMFirstStackingPredictor:
    """LSTM 모델을 먼저 학습하여 분석한 후, 그 상승 확률 예측 결과를 피처로 사용하여 XGBoost를 학습시키는 스태킹 예측기"""

    def __init__(self, sequence_length: int = 20, scanner_mode: bool = False):
        self.sequence_length = sequence_length
        self.scanner_mode = scanner_mode
        self.lstm = LSTMPredictor(sequence_length=sequence_length, scanner_mode=scanner_mode)
        self.xgb = NewXGBoostPredictor(scanner_mode=scanner_mode)
        self.is_trained = False
        self.training_metrics: dict[str, Any] = {}
        self.feature_importances_: pd.Series | None = None

    def train(self, df: pd.DataFrame, include_sentiment: bool = True) -> dict[str, Any]:
        t_total = time.time()

        # Check if PyTorch or TensorFlow is available
        fw = self.lstm.available_framework()
        if fw is None:
            log.warning("PyTorch/TensorFlow 미설치로 인해 LSTM 학습을 생략하고 XGBoost 단독으로 학습합니다.")
            xgb_metrics = self.xgb.train(df, include_sentiment=include_sentiment)
            if "error" in xgb_metrics:
                return xgb_metrics
            self.is_trained = True
            self.training_metrics = {
                "model_type": "LSTM-XGB Stacking (XGB Fallback)",
                "lstm_metrics": {"error": "Framework not available"},
                "xgb_metrics": xgb_metrics,
                "n_samples": xgb_metrics.get("n_samples", 0),
                "n_features": len(self.xgb.feature_cols or []),
                "cv_accuracy_mean": xgb_metrics.get("cv_accuracy_mean", 0),
                "elapsed_sec": round(time.time() - t_total, 1),
            }
            return self.training_metrics

        # Step 1: 깨끗한 OOF LSTM 확률 생성
        orig_feature_cols = get_feature_columns(df, include_sentiment)
        feature_cols = orig_feature_cols + ["LSTM_Proba"]
        oof_lstm_probs = _compute_oof_lstm_proba(
            df,
            sequence_length=self.sequence_length,
            include_sentiment=include_sentiment,
            n_splits=5,
            scanner_mode=self.scanner_mode,
        )

        df_oof = df.copy()
        df_oof["LSTM_Proba"] = oof_lstm_probs

        # Step 2: XGBoost CV 평가를 위한 OOF 메타 피처 학습
        xgb_metrics = self.xgb.train(
            df_oof,
            include_sentiment=include_sentiment,
            feature_cols=feature_cols,
            cv_only=True,
        )
        if "error" in xgb_metrics:
            log.warning(f"NewXGBoost 학습 실패: {xgb_metrics.get('error')}")
            return xgb_metrics

        # Step 3: 최종 LSTM은 전체 데이터로 재학습
        lstm_metrics = self.lstm.train(df, include_sentiment=include_sentiment)
        if "error" in lstm_metrics:
            log.warning(f"LSTM 학습 실패: {lstm_metrics.get('error')}")
            return lstm_metrics

        # Step 4: 최종 XGBoost는 전체 데이터의 최종 LSTM 피처로 재학습
        df_full = df.copy()
        df_full["LSTM_Proba"] = predict_lstm_history_proba(self.lstm, df)
        full_xgb_metrics = self.xgb.fit_full_data(
            df_full, include_sentiment=include_sentiment, feature_cols=feature_cols
        )
        if "error" in full_xgb_metrics:
            log.warning(f"최종 XGBoost 재학습 실패: {full_xgb_metrics.get('error')}")
            return full_xgb_metrics
        xgb_metrics["full_train_accuracy"] = full_xgb_metrics.get("train_accuracy", 0)

        self.is_trained = True
        self.feature_importances_ = self.xgb.feature_importances_

        self.training_metrics = {
            "model_type": "LSTM-XGB Stacking",
            "lstm_metrics": lstm_metrics,
            "xgb_metrics": xgb_metrics,
            "n_samples": xgb_metrics.get("n_samples", 0),
            "n_features": len(feature_cols),
            "cv_accuracy_mean": xgb_metrics.get("cv_accuracy_mean", 0),
            "elapsed_sec": round(time.time() - t_total, 1),
        }
        return self.training_metrics

    def predict(self, df: pd.DataFrame) -> dict[str, Any]:
        if not self.is_trained:
            return {"error": "모델 미학습"}

        # Check if PyTorch or TensorFlow is available
        fw = self.lstm.available_framework()
        if fw is None:
            p_xgb = self.xgb.predict_proba(df)
            if p_xgb is None:
                return {"error": "XGBoost 예측 실패"}
            result = _build_result(p_xgb, "LSTM-XGB Stacking (XGB Fallback)")
            result["ensemble_detail"] = {
                "p_lstm": None,
                "p_xgb": round(p_xgb, 4),
                "w_lstm": 0.0,
                "w_xgb": 1.0,
                "complexity": 0.0,
                "regime": "fallback",
            }
            return result

        # Step 1: LSTM 예측 확률 산출
        p_lstm_series = predict_lstm_history_proba(self.lstm, df)
        if len(p_lstm_series) == 0:
            return {"error": "LSTM 예측 실패"}

        # Step 2: 데이터프레임에 피처 추가
        df_with_lstm = df.copy()
        df_with_lstm["LSTM_Proba"] = p_lstm_series

        # Step 3: XGBoost 최종 예측
        p_xgb = self.xgb.predict_proba(df_with_lstm)
        if p_xgb is None:
            return {"error": "XGBoost 예측 실패"}

        last_lstm = float(p_lstm_series.iloc[-1]) if len(p_lstm_series) > 0 else 0.5
        result = _build_result(p_xgb, "LSTM-XGB Stacking")
        result["ensemble_detail"] = {
            "p_lstm": round(last_lstm, 4),
            "p_xgb": round(p_xgb, 4),
            "w_lstm": 0.0,
            "w_xgb": 1.0,
            "complexity": 0.0,
            "regime": "stacking",
        }
        return result
