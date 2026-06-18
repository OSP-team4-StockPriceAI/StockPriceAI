import random
import uuid
from locust import HttpUser, task, between

COMMON_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "NFLX", "DIS", "JPM"]

class StockPriceAIUser(HttpUser):
    wait_time = between(1, 3)  # 각 요청 사이에 1~3초 대기
    
    def on_start(self):
        """유저 테스트 시작 시 호출: 회원가입 후 로그인을 수행하여 JWT 토큰을 발급받습니다."""
        self.email = f"loadtest_{uuid.uuid4().hex[:8]}@example.com"
        self.password = "testpass123!"
        self.headers = {}
        
        # 1. 회원가입
        register_payload = {
            "email": self.email,
            "password": self.password
        }
        with self.client.post(
            "/api/v1/auth/register", 
            json=register_payload, 
            catch_response=True
        ) as response:
            if response.status_code == 201:
                response.success()
            else:
                response.failure(f"Registration failed: {response.text}")
                return
                
        # 2. 로그인 및 토큰 저장
        login_payload = {
            "email": self.email,
            "password": self.password
        }
        with self.client.post(
            "/api/v1/auth/login", 
            json=login_payload, 
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                access_token = data.get("access_token")
                self.headers = {"Authorization": f"Bearer {access_token}"}
                response.success()
            else:
                response.failure(f"Login failed: {response.text}")

    @task(2)
    def health_check(self):
        """간단한 헬스체크 엔드포인트를 호출합니다."""
        self.client.get("/health")

    @task(5)
    def get_stock_info(self):
        """종목 상세 정보를 조회합니다 (공개 API, Redis 캐시 조회 또는 ML 호출)."""
        ticker = random.choice(COMMON_TICKERS)
        self.client.get(f"/api/v1/stocks/{ticker}")

    @task(3)
    def get_predictions(self):
        """종목 예측 이력을 조회합니다 (인증 필요, DB 조회 또는 ML 호출)."""
        if not self.headers:
            return  # 로그인이 실패한 경우 건너뜀
        ticker = random.choice(COMMON_TICKERS)
        self.client.get(f"/api/v1/predictions/{ticker}", headers=self.headers)
