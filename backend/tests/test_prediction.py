from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.orm import Session

from app.models.prediction import Prediction
from app.services.prediction import get_or_fetch_predictions


@pytest.mark.asyncio
async def test_get_predictions_already_in_db(setup_db, db_session: Session):
    """DB에 이미 예측 결과가 저장되어 있는 경우, ML 호출 없이 바로 리턴하는지 확인합니다."""
    ticker = "AAPL"
    
    # 1. 테스트용 예측 데이터를 DB에 선생성
    pred = Prediction(
        ticker=ticker,
        signal="BUY",
        up_prob=0.85,
        model_type="Ensemble",
        complexity=2.0,
        xgb_weight=0.6,
        lstm_weight=0.4
    )
    db_session.add(pred)
    db_session.commit()
    
    # 2. _call_ml_predict가 호출되지 않는지 Mocking하여 검증
    with patch("app.services.prediction._call_ml_predict", new_callable=AsyncMock) as mock_call:
        results = await get_or_fetch_predictions(ticker=ticker, limit=10, db=db_session)
        
        assert len(results) == 1
        assert results[0].ticker == ticker
        assert results[0].signal == "BUY"
        assert results[0].up_prob == 0.85
        mock_call.assert_not_called()  # 캐시 히트이므로 호출되지 않아야 함


@pytest.mark.asyncio
async def test_get_predictions_cache_miss_calls_ml(setup_db, db_session: Session):
    """DB에 데이터가 없을 때, ML 서비스를 호출하고 그 결과를 DB에 저장한 뒤 리턴하는지 검증합니다."""
    ticker = "TSLA"
    
    mock_ml_response = {
        "ticker": ticker,
        "signal": "SELL",
        "up_probability": 0.15,
        "down_probability": 0.85,
        "confidence": 0.85,
        "model": "Ensemble",
        "ensemble_detail": {
            "complexity": 3.0,
            "w_xgb": 0.7,
            "w_lstm": 0.3
        },
        "training_metrics": {},
        "technical_summary": {}
    }
    
    with patch("app.services.prediction._call_ml_predict", new_callable=AsyncMock, return_value=mock_ml_response) as mock_call:
        results = await get_or_fetch_predictions(ticker=ticker, limit=10, db=db_session)
        
        assert len(results) == 1
        assert results[0].ticker == ticker
        assert results[0].signal == "SELL"
        assert results[0].up_prob == 0.15
        mock_call.assert_called_once_with(ticker)
        
        # DB에 저장되었는지 확인
        db_rows = db_session.query(Prediction).filter(Prediction.ticker == ticker).all()
        assert len(db_rows) == 1
        assert db_rows[0].signal == "SELL"


@pytest.mark.asyncio
async def test_get_predictions_ml_down_no_history_raises_error(setup_db, db_session: Session):
    """ML 서비스가 꺼져있고 DB에 과거 데이터도 없을 경우, 예외가 정상적으로 상위로 전달되는지 확인합니다."""
    ticker = "NVDA"
    
    with patch("app.services.prediction._call_ml_predict", new_callable=AsyncMock, side_effect=Exception("ML service offline")):
        with pytest.raises(Exception, match="ML service offline"):
            await get_or_fetch_predictions(ticker=ticker, limit=10, db=db_session)


@pytest.mark.asyncio
async def test_get_predictions_ml_down_with_history_graceful_fallback(setup_db, db_session: Session):
    """ML 서비스가 꺼져있지만 DB에 오래된 과거 데이터가 있는 경우, 503 크래시 대신 과거의 데이터(Stale)를 정상 리턴하는지 검증합니다."""
    ticker = "NVDA"
    
    # 1. 오래된 과거 예측 데이터를 DB에 미리 넣어둠
    stale_pred = Prediction(
        ticker=ticker,
        signal="HOLD",
        up_prob=0.50,
        model_type="Ensemble",
        complexity=1.0,
        xgb_weight=0.5,
        lstm_weight=0.5
    )
    db_session.add(stale_pred)
    db_session.commit()
    
    # 2. ML 서비스 호출이 실패하도록 에러를 뱉는 mock 설정
    with patch("app.services.prediction._call_ml_predict", new_callable=AsyncMock, side_effect=Exception("ML service offline")):
        results = await get_or_fetch_predictions(ticker=ticker, limit=10, db=db_session)
        
        # 3. 크래시 없이 과거 데이터가 성공적으로 리턴되었는지 검증
        assert len(results) == 1
        assert results[0].ticker == ticker
        assert results[0].signal == "HOLD"
        assert results[0].up_prob == 0.50
