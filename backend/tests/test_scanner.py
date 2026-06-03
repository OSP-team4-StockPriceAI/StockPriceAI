import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.scan import ScanJob, ScanResult
from app.services.scanner import create_scan_job, get_scan_job, get_scan_results, list_scan_jobs


@pytest.mark.asyncio
async def test_create_scan_job_success(setup_db, db_session: Session):
    """스캔 작업 생성이 성공적으로 DB에 등록되고 ML 서비스를 정상 호출하는지 확인합니다."""
    user_id = uuid.uuid4()
    sector = "Technology"
    mock_ml_id = str(uuid.uuid4())

    with patch("app.services.scanner.get_http_client") as mock_client_getter:
        mock_client = AsyncMock()
        mock_client_getter.return_value = mock_client
        
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = lambda: {"job_id": mock_ml_id, "status": "queued"}
        mock_client.post.return_value = mock_response

        job = await create_scan_job(user_id=user_id, sector=sector, db=db_session)

        # 검증: ML이 반환한 ID가 PK로 정상 등록되었는지 확인
        assert str(job.id) == mock_ml_id
        assert job.sector == sector
        assert job.status == "pending"
        
        # ML 서비스 호출 확인
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args[1]
        assert sector in call_args["json"]["sector"]


@pytest.mark.asyncio
async def test_create_scan_job_ml_offline(setup_db, db_session: Session):
    """ML 서비스가 꺼져있을 경우, 503 에러가 정상 반환되고 DB에 등록되지 않는지 확인합니다."""
    user_id = uuid.uuid4()
    sector = "Technology"

    with patch("app.services.scanner.get_http_client") as mock_client_getter:
        mock_client = AsyncMock()
        mock_client_getter.return_value = mock_client
        mock_client.post.side_effect = httpx.RequestError("ML offline")

        with pytest.raises(HTTPException) as exc_info:
            await create_scan_job(user_id=user_id, sector=sector, db=db_session)

        assert exc_info.value.status_code == 503
        assert "ML service unavailable" in exc_info.value.detail

        # DB 검증: 등록되지 않고 깨끗하게 복구되었는지 확인
        db_job = db_session.query(ScanJob).filter(ScanJob.user_id == user_id).first()
        assert db_job is None


@pytest.mark.asyncio
async def test_get_scan_job_ownership_success(setup_db, db_session: Session):
    """자신이 생성한 스캔 작업을 성공적으로 조회하는지 검증합니다."""
    user_id = uuid.uuid4()
    
    job = ScanJob(user_id=user_id, sector="Energy", status="pending")
    db_session.add(job)
    db_session.commit()

    with patch("app.services.scanner.sync_job_status_from_ml") as mock_sync:
        mock_sync.return_value = job
        result = await get_scan_job(job_id=job.id, user_id=user_id, db=db_session)
        assert result.id == job.id
        assert result.sector == "Energy"


@pytest.mark.asyncio
async def test_get_scan_job_ownership_forbidden(setup_db, db_session: Session):
    """타인의 스캔 작업을 조회하려 할 때 404 에러를 뱉는지(보안 검증) 확인합니다."""
    owner_id = uuid.uuid4()
    attacker_id = uuid.uuid4()
    
    job = ScanJob(user_id=owner_id, sector="Energy", status="pending")
    db_session.add(job)
    db_session.commit()

    with pytest.raises(HTTPException) as exc_info:
        with patch("app.services.scanner.sync_job_status_from_ml") as mock_sync:
            mock_sync.return_value = job
            await get_scan_job(job_id=job.id, user_id=attacker_id, db=db_session)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_scan_results_ownership_success(setup_db, db_session: Session):
    """자신이 생성한 스캔 작업의 매수/매도 분석 결과를 조회하는지 검증합니다."""
    user_id = uuid.uuid4()
    
    job = ScanJob(user_id=user_id, sector="Healthcare", status="completed")
    db_session.add(job)
    db_session.commit()

    res = ScanResult(
        job_id=job.id,
        ticker="PFE",
        composite_score=0.85,
        up_prob=0.88,
        signal="BUY",
        sector="Healthcare"
    )
    db_session.add(res)
    db_session.commit()

    results = await get_scan_results(job_id=job.id, user_id=user_id, db=db_session)
    assert len(results) == 1
    assert results[0].ticker == "PFE"
    assert results[0].signal == "BUY"


@pytest.mark.asyncio
async def test_get_scan_results_ownership_forbidden(setup_db, db_session: Session):
    """타인의 스캔 결과를 탈취하여 조회하려 할 때 404 에러로 격리 방어하는지 검증합니다."""
    owner_id = uuid.uuid4()
    attacker_id = uuid.uuid4()
    
    job = ScanJob(user_id=owner_id, sector="Healthcare", status="completed")
    db_session.add(job)
    db_session.commit()

    with pytest.raises(HTTPException) as exc_info:
        await get_scan_results(job_id=job.id, user_id=attacker_id, db=db_session)

    assert exc_info.value.status_code == 404


def test_list_scan_jobs(setup_db, db_session: Session):
    """사용자가 생성했던 최근 스캔 목록들을 최신순으로 가져오는지 검증합니다."""
    user_id = uuid.uuid4()
    
    # SQLite 등에서 func.now() 동시 생성으로 인한 정렬 불확정성 방지
    now = datetime.utcnow()
    job1 = ScanJob(user_id=user_id, sector="Tech", status="completed", created_at=now - timedelta(minutes=10))
    job2 = ScanJob(user_id=user_id, sector="Bio", status="pending", created_at=now)
    db_session.add_all([job1, job2])
    db_session.commit()

    jobs = list_scan_jobs(user_id=user_id, db=db_session)
    assert len(jobs) == 2
    # 최신순 정렬 확인
    assert jobs[0].sector == "Bio"
    assert jobs[1].sector == "Tech"
