"""
LSTM-XGBoost Stacking 예측 모듈 (Production Optimized)
구조:
  LSTMFirstStackingPredictor  — LSTM OOF 확률 피처 + 트리 모델의 결합
  NewXGBoostPredictor         — 스케일링 오버헤드를 제거한 고속 메타 모델
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

from ..core.config import (
    XGBOOST,
    XGBOOST_SCANNER,
)
from .predictor import (
    LSTMPredictor,
    XGBoostPredictor,
    _auto_params,
    _build_result,
    get_feature_columns,
    prepare_training_data,
)

warnings.filterwarnings("ignore")
log = logging.getLogger("stockai.ml")


def predict_lstm_history_proba(lstm_pred: LSTMPredictor, df: pd.DataFrame) -> pd.Series:
    """학습된 LSTMPredictor를 사용하여 입력 데이터 전체에 대한 LSTM 예측 확률을 생성합니다."""
    if not lstm_pred.is_trained or lstm_pred.feature_cols is None:
        return pd.Series(0.5, index=df.index)

    SEQ = lstm_pred.sequence_length
    feature_cols = lstm_pred.feature_cols

    # 메모리 절약을 위한 float32 고정 및 고속 결측치 처리
    work = df[feature_cols].ffill().bfill().fillna(0).replace([np.inf, -np.inf], 0).clip(-10, 10)
    
    if len(work) < SEQ:
        return pd.Series(0.5, index=df.index)

    X = work.values.astype(np.float32)
    X_sc = lstm_pred.scaler.transform(X)
    
    probs = _predict_lstm_on_scaled_features(lstm_pred, X_sc, SEQ)
    return pd.Series(probs, index=df.index)


def _predict_lstm_on_scaled_features(
    lstm_pred: LSTMPredictor,
    X_sc: npt.NDArray[np.float32],
    sequence_length: int,
) -> npt.NDArray[np.float64]:
    """파이썬 루프를 제거하고 Numpy 메모리 뷰를 활용한 초고속 시퀀스 생성 및 추론"""
    probs = np.full(len(X_sc), 0.5, dtype=np.float64)
    
    if len(X_sc) < sequence_length:
        return probs
    
    # [최적화] np.lib.stride_tricks 활용: 메모리 복사 없이 C 레벨 속도로 3D 시퀀스 뷰 생성
    # X_sc shape: (N, F) -> sliding_window shape: (N - SEQ + 1, F, SEQ)
    windowed = np.lib.stride_tricks.sliding_window_view(X_sc, window_shape=sequence_length, axis=0)
    # 텐서 배열을 LSTM 입력 규격인 (Batch, Seq, Feature)로 변경
    X_seq = np.transpose(windowed, (0, 2, 1)).copy() 
    
    if lstm_pred.framework == "pytorch":
        import torch
        device = next(lstm_pred.model.parameters()).device
        lstm_pred.model.eval()
        with torch.no_grad():
            t = torch.tensor(X_seq, dtype=torch.float32, device=device)
            preds = lstm_pred.model(t).cpu().numpy().flatten()
            probs[sequence_length - 1:] = preds
    else:
        preds = lstm_pred.model.predict(X_seq, verbose=0).flatten()
        probs[sequence_length - 1:] = preds
    
    return probs


def _compute_oof_lstm_proba(
    df: pd.DataFrame,
    sequence_length: int,
    include_sentiment: bool,
    n_splits: int = 5,
    scanner_mode: bool = False,
) -> pd.Series:
    """OOF LSTM 확률 생성기"""
    oof = pd.Series(0.5, index=df.index, dtype=float)
    if len(df) < sequence_length + 20:
        return oof

    n_splits = min(n_splits, max(2, len(df) // (sequence_length + 1)))
    if n_splits < 2:
        return oof

    feature_cols = get_feature_columns(df, include_sentiment)
    
    work = df[feature_cols].ffill().bfill().fillna(0).replace([np.inf, -np.inf], 0).clip(-10, 10)
    X_full = work.values.astype(np.float32)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, val_idx in tscv.split(df):
        df_tr = df.iloc[train_idx]
        fold_lstm = LSTMPredictor(sequence_length=sequence_length, scanner_mode=scanner_mode)
        fold_metrics = fold_lstm.train(df_tr, include_sentiment=include_sentiment)
        if "error" in fold_metrics:
            continue

        start_idx = max(0, val_idx[0] - sequence_length + 1)
        end_idx = val_idx[-1] + 1
        
        X_sc = fold_lstm.scaler.transform(X_full[start_idx:end_idx])
        fold_probs = _predict_lstm_on_scaled_features(fold_lstm, X_sc, sequence_length)
        
        val_start_offset = val_idx[0] - start_idx
        val_end_offset = val_start_offset + len(val_idx)
        val_index = df.iloc[val_idx].index
        oof.loc[val_index] = fold_probs[val_start_offset:val_end_offset]

    return oof


class NewXGBoostPredictor(XGBoostPredictor):
    """트리 모델의 특성을 반영하여 불필요한 Scaling 오버헤드를 제거한 고성능 예측기"""

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
        except Exception:
            return self._train_sklearn(df, include_sentiment, feature_cols=feature_cols)

        xgb_cfg = XGBOOST_SCANNER if self.scanner_mode else XGBOOST
        self.feature_cols = feature_cols if feature_cols is not None else get_feature_columns(df, include_sentiment)

        raw_len = len(df)
        ap = _auto_params(raw_len)
        max_samp = ap.get("max_samples")
        n_est_cv = max(80, ap["n_estimators_xgb"] - 100) if not self.scanner_mode else 100
        n_est_fin = ap["n_estimators_xgb"] if not self.scanner_mode else 150
        n_splits_ = ap["n_splits"] if not self.scanner_mode else 3

        X, y, _ = prepare_training_data(df, self.feature_cols, max_samples=max_samp)
        if X is None or y is None:
            return {"error": "학습 데이터 부족 (최소 60일 필요)"}

        # XGBoost는 float32를 네이티브로 처리합니다. 스케일링 제거.
        X = X.astype(np.float32)

        log.info(f"New XGBoost 학습: 데이터={raw_len}일, 피처={len(self.feature_cols)}개, CV={n_splits_}fold")

        t0 = time.time()
        tscv = TimeSeriesSplit(n_splits=n_splits_)
        cv_scores = []
        oof_proba = np.full(len(y), 0.5)

        for tr_idx, val_idx in tscv.split(X):
            Xtr, Xvl = X[tr_idx], X[val_idx]
            ytr, yvl = y[tr_idx], y[val_idx]

            m = xgb.XGBClassifier(
                n_estimators=n_est_cv, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5, gamma=1,
                reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss", random_state=42,
                n_jobs=xgb_cfg["nthread"], verbosity=0, device=xgb_cfg["device"],
                tree_method=xgb_cfg["tree_method"], max_bin=xgb_cfg["max_bin"], grow_policy=xgb_cfg["grow_policy"],
            )
            m.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], verbose=False)
            oof_proba[val_idx] = m.predict_proba(Xvl)[:, 1]
            cv_scores.append(accuracy_score(yvl, m.predict(Xvl)))

        self._cv_proba = oof_proba
        log.info(f"New XGBoost CV 평균 정확도: {float(np.mean(cv_scores)):.3f} ({time.time()-t0:.1f}s)")

        if not cv_only:
            self.model = xgb.XGBClassifier(
                n_estimators=n_est_fin, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5, gamma=1,
                reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss", random_state=42,
                n_jobs=xgb_cfg["nthread"], verbosity=0, device=xgb_cfg["device"],
                tree_method=xgb_cfg["tree_method"], max_bin=xgb_cfg["max_bin"], grow_policy=xgb_cfg["grow_policy"],
            )
            self.model.fit(X, y, verbose=False)
            self.is_trained = True
            self.feature_importances_ = pd.Series(self.model.feature_importances_, index=self.feature_cols).sort_values(ascending=False)
            train_accuracy = float(accuracy_score(y, self.model.predict(X)))
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
        except Exception:
            return self._train_sklearn(df, include_sentiment, feature_cols=feature_cols)

        xgb_cfg = XGBOOST_SCANNER if self.scanner_mode else XGBOOST
        self.feature_cols = feature_cols if feature_cols is not None else get_feature_columns(df, include_sentiment)

        ap = _auto_params(len(df))
        X, y, _ = prepare_training_data(df, self.feature_cols, max_samples=ap.get("max_samples"))
        if X is None or y is None:
            return {"error": "학습 데이터 부족 (최소 60일 필요)"}

        X = X.astype(np.float32)

        self.model = xgb.XGBClassifier(
            n_estimators=ap["n_estimators_xgb"] if not self.scanner_mode else 150,
            max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, gamma=1, reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
            random_state=42, n_jobs=xgb_cfg["nthread"], verbosity=0, device=xgb_cfg["device"],
            tree_method=xgb_cfg["tree_method"], max_bin=xgb_cfg["max_bin"], grow_policy=xgb_cfg["grow_policy"],
        )
        self.model.fit(X, y, verbose=False)
        self.is_trained = True

        self.feature_importances_ = pd.Series(self.model.feature_importances_, index=self.feature_cols).sort_values(ascending=False)
        self.training_metrics = {
            "train_accuracy": float(accuracy_score(y, self.model.predict(X))),
            "model_type": "NewXGBoost (full)",
            "n_features": len(self.feature_cols),
            "n_samples": len(y),
            "n_samples_total": len(df),
        }
        return self.training_metrics

    def _train_sklearn(
        self,
        df: pd.DataFrame,
        include_sentiment: bool,
        feature_cols: list[str] | None = None,
    ) -> dict[str, Any]:
        from sklearn.ensemble import GradientBoostingClassifier
        log.warning("XGBoost 로드 실패 → GradientBoosting 폴백 (Scaling 미적용)")
        
        self.feature_cols = feature_cols if feature_cols is not None else get_feature_columns(df, include_sentiment)
        X, y, _ = prepare_training_data(df, self.feature_cols)
        if X is None or y is None:
            return {"error": "학습 데이터 부족"}

        self.model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        self.feature_importances_ = pd.Series(self.model.feature_importances_, index=self.feature_cols).sort_values(ascending=False)

        self.training_metrics = {
            "train_accuracy": float(accuracy_score(y, self.model.predict(X))),
            "model_type": "GradientBoosting (sklearn)",
            "n_features": len(self.feature_cols),
            "n_samples": len(y),
        }
        return self.training_metrics


    def predict_proba(self, df: pd.DataFrame) -> float | None:
        """스케일링 없이 float32로 직접 예측 (부모 클래스의 scaler 의존성 제거)"""
        if not self.is_trained or self.feature_cols is None:
            return None
        try:
            latest = df[self.feature_cols].iloc[-1:]
            latest = latest.ffill().bfill().fillna(0).replace([np.inf, -np.inf], 0).clip(-10, 10)
            X = latest.values.astype(np.float32)
            return float(self.model.predict_proba(X)[0, 1])
        except Exception:
            return None


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