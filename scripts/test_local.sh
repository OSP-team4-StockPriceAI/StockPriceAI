#!/usr/bin/env bash
# 로컬 통합 테스트 자동화 스크립트
# 사용법: ./scripts/test_local.sh [--full] [--skip-ml]
#   --full     : ML 예측 테스트 포함 (30초~2분 소요)
#   --skip-ml  : ML 서비스 테스트 전체 건너뜀

set -euo pipefail

BACKEND="http://localhost:8000"
ML="http://localhost:8001"
NGINX="http://localhost"
FULL=false
SKIP_ML=false

for arg in "$@"; do
  case $arg in
    --full)    FULL=true ;;
    --skip-ml) SKIP_ML=true ;;
  esac
done

# ── 색상 ──────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

PASS=0; FAIL=0

pass() { echo -e "  ${GREEN}✓${RESET} $1"; PASS=$((PASS+1)); }
fail() { echo -e "  ${RED}✗${RESET} $1"; FAIL=$((FAIL+1)); }
section() { echo -e "\n${CYAN}${BOLD}▶ $1${RESET}"; }
skip() { echo -e "  ${YELLOW}–${RESET} $1 (건너뜀)"; }

# ── 헬퍼 ──────────────────────────────────────────────
# 응답 코드 확인
check_status() {
  local label="$1" url="$2" method="${3:-GET}" data="${4:-}" expected="${5:-200}"
  local args=(-s -o /dev/null -w "%{http_code}" --max-time 15)
  [[ $method != "GET" ]] && args+=(-X "$method")
  [[ -n $data ]] && args+=(-H "Content-Type: application/json" -d "$data")
  [[ -n ${TOKEN:-} ]] && args+=(-H "Authorization: Bearer $TOKEN")

  local code
  code=$(curl "${args[@]}" "$url" 2>/dev/null || echo "000")
  if [[ $code == "$expected" ]]; then
    pass "$label → HTTP $code"
  else
    fail "$label → 기대 $expected, 실제 $code"
  fi
}

# JSON 필드 값 추출
json_get() {
  python3 -c "import sys,json; d=json.load(sys.stdin); print($1)" 2>/dev/null
}

# ── 사전 확인 ─────────────────────────────────────────
section "0. 사전 확인"

if ! command -v docker &>/dev/null; then
  echo -e "${RED}docker 명령어를 찾을 수 없습니다. Docker를 설치하세요.${RESET}"
  exit 1
fi
if ! command -v curl &>/dev/null; then
  echo -e "${RED}curl 명령어를 찾을 수 없습니다.${RESET}"
  exit 1
fi
pass "docker, curl 확인"

# ── 3. 헬스체크 ───────────────────────────────────────
section "1. 헬스체크"

# 백엔드 헬스
resp=$(curl -s --max-time 10 "$BACKEND/health" 2>/dev/null || echo "{}")
if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok'" 2>/dev/null; then
  pass "Backend /health → status: ok"
else
  fail "Backend /health → 응답 이상 ($resp)"
fi

# ML 헬스
if [[ $SKIP_ML == false ]]; then
  resp=$(curl -s --max-time 10 "$ML/health" 2>/dev/null || echo "{}")
  if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok'" 2>/dev/null; then
    pass "ML /health → status: ok"
  else
    fail "ML /health → 응답 이상 ($resp)"
  fi
fi

# Nginx 라우팅
resp=$(curl -s --max-time 10 "$NGINX/api/health" 2>/dev/null || echo "{}")
if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok'" 2>/dev/null; then
  pass "Nginx /api/health → backend 라우팅 정상"
else
  fail "Nginx /api/health → 응답 이상 ($resp)"
fi

if [[ $SKIP_ML == false ]]; then
  resp=$(curl -s --max-time 10 "$NGINX/ml/health" 2>/dev/null || echo "{}")
  if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok'" 2>/dev/null; then
    pass "Nginx /ml/health → ml 라우팅 정상"
  else
    fail "Nginx /ml/health → 응답 이상 ($resp)"
  fi
fi

# Nginx → Backend 라우팅: /api/v1/stocks 경로가 이중 prefix 없이 전달되는지 확인
# NEXT_PUBLIC_API_URL에 /api가 포함되면 /api/api/v1/... 가 되어 404 발생
STOCKS_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$NGINX/api/v1/stocks/AAPL" 2>/dev/null || echo "000")
if [[ $STOCKS_CODE == "200" || $STOCKS_CODE == "401" || $STOCKS_CODE == "422" ]]; then
  pass "Nginx /api/v1/stocks 라우팅 정상 → HTTP $STOCKS_CODE (404 아님)"
else
  fail "Nginx /api/v1/stocks 라우팅 오류 → HTTP $STOCKS_CODE (NEXT_PUBLIC_API_URL에 /api 포함 여부 확인)"
fi

# PostgreSQL
if docker compose -f "$(dirname "$0")/../docker-compose.yml" exec -T postgres \
     psql -U stockai -d stockai -c "\dt" 2>/dev/null | grep -q "users"; then
  pass "PostgreSQL 테이블 존재 확인 (users 포함)"
else
  fail "PostgreSQL 테이블 확인 실패"
fi

# Redis
if docker compose -f "$(dirname "$0")/../docker-compose.yml" exec -T redis \
     redis-cli ping 2>/dev/null | grep -q "PONG"; then
  pass "Redis PING → PONG"
else
  fail "Redis PING 실패"
fi

# ── 4. 백엔드 API ──────────────────────────────────────
section "2. 백엔드 API 테스트"

TEST_EMAIL="autotest_$(date +%s)@example.com"
TEST_PASS="testpass123"

# 4-1. 회원가입
REGISTER=$(curl -s --max-time 15 -X POST "$BACKEND/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d "{\"email\": \"$TEST_EMAIL\", \"password\": \"$TEST_PASS\"}" 2>/dev/null || echo "{}")
if echo "$REGISTER" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'id' in d" 2>/dev/null; then
  pass "회원가입 → id 반환"
else
  fail "회원가입 실패: $REGISTER"
fi

# 4-2. 로그인 (JWT 발급)
LOGIN=$(curl -s --max-time 15 -X POST "$BACKEND/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\": \"$TEST_EMAIL\", \"password\": \"$TEST_PASS\"}" 2>/dev/null || echo "{}")
TOKEN=$(echo "$LOGIN" | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])" 2>/dev/null || echo "")
if [[ -n $TOKEN ]]; then
  pass "로그인 → access_token 발급"
else
  fail "로그인 실패 (토큰 없음): $LOGIN"
fi

# 4-3. 내 정보 조회
ME=$(curl -s --max-time 10 "$BACKEND/api/v1/auth/me" \
  -H "Authorization: Bearer $TOKEN" 2>/dev/null || echo "{}")
if echo "$ME" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('email')" 2>/dev/null; then
  pass "내 정보 조회 → email 반환"
else
  fail "내 정보 조회 실패: $ME"
fi

# 4-4. 관심종목 CRUD
ADD=$(curl -s --max-time 10 -X POST "$BACKEND/api/v1/watchlist" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "memo": "자동화 테스트"}' 2>/dev/null || echo "{}")
if echo "$ADD" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('ticker')=='AAPL'" 2>/dev/null; then
  pass "관심종목 추가 (AAPL)"
else
  fail "관심종목 추가 실패: $ADD"
fi

LIST=$(curl -s --max-time 10 "$BACKEND/api/v1/watchlist" \
  -H "Authorization: Bearer $TOKEN" 2>/dev/null || echo "[]")
if echo "$LIST" | python3 -c "import sys,json; d=json.load(sys.stdin); assert any(x.get('ticker')=='AAPL' for x in d)" 2>/dev/null; then
  pass "관심종목 목록 조회 (AAPL 포함)"
else
  fail "관심종목 목록 조회 실패: $LIST"
fi

PATCH=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 -X PATCH "$BACKEND/api/v1/watchlist/AAPL" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"memo": "수정된 메모"}' 2>/dev/null || echo "000")
if [[ $PATCH == "200" ]]; then
  pass "관심종목 메모 수정 → HTTP 200"
else
  fail "관심종목 메모 수정 → HTTP $PATCH"
fi

DEL=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 -X DELETE "$BACKEND/api/v1/watchlist/AAPL" \
  -H "Authorization: Bearer $TOKEN" 2>/dev/null || echo "000")
if [[ $DEL == "204" ]]; then
  pass "관심종목 삭제 → HTTP 204"
else
  fail "관심종목 삭제 → HTTP $DEL"
fi

# 4-5. 스캔 작업 생성
JOB=$(curl -s --max-time 15 -X POST "$BACKEND/api/v1/scanner/jobs" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"sector": "Technology"}' 2>/dev/null || echo "{}")
BACKEND_JOB_ID=$(echo "$JOB" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null || echo "")
if [[ -n $BACKEND_JOB_ID ]]; then
  pass "스캔 작업 생성 → job_id: $BACKEND_JOB_ID"
else
  fail "스캔 작업 생성 실패: $JOB"
fi

# ── 5. ML 서비스 테스트 ────────────────────────────────
if [[ $SKIP_ML == true ]]; then
  section "3. ML 서비스 테스트"
  skip "ML 서비스 테스트 전체 (--skip-ml)"
else
  section "3. ML 서비스 테스트"

  # 5-2. 기술적 지표
  TECH=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 "$ML/api/v1/technical/AAPL" 2>/dev/null || echo "000")
  if [[ $TECH == "200" ]]; then
    pass "기술적 지표 AAPL → HTTP 200"
  else
    fail "기술적 지표 AAPL → HTTP $TECH"
  fi

  # 5-3. 감성 분석
  SENT=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 "$ML/api/v1/sentiment/AAPL" 2>/dev/null || echo "000")
  if [[ $SENT == "200" ]]; then
    pass "감성 분석 AAPL → HTTP 200"
  else
    fail "감성 분석 AAPL → HTTP $SENT"
  fi

  # 5-4. 배치 스캔 (소규모)
  SCAN=$(curl -s --max-time 30 -X POST "$ML/api/v1/scanner/start" \
    -H "Content-Type: application/json" \
    -d '{"tickers":["AAPL","MSFT"],"max_workers":1,"force_refresh":false,"period_days":400}' \
    2>/dev/null || echo "{}")
  SCAN_JOB_ID=$(echo "$SCAN" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])" 2>/dev/null || echo "")
  if [[ -n $SCAN_JOB_ID ]]; then
    pass "배치 스캔 시작 → job_id: $SCAN_JOB_ID"
  else
    fail "배치 스캔 시작 실패: $SCAN"
  fi

  # 스캔 상태 확인
  if [[ -n $SCAN_JOB_ID ]]; then
    STATUS=$(curl -s --max-time 10 "$ML/api/v1/scanner/status/$SCAN_JOB_ID" 2>/dev/null || echo "{}")
    if echo "$STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'status' in d" 2>/dev/null; then
      pass "스캔 상태 조회 → $(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','?'))" 2>/dev/null)"
    else
      fail "스캔 상태 조회 실패: $STATUS"
    fi
  fi

  # 캐시 통계
  CACHE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$ML/api/v1/scanner/cache/stats" 2>/dev/null || echo "000")
  if [[ $CACHE == "200" ]]; then
    pass "캐시 통계 → HTTP 200"
  else
    fail "캐시 통계 → HTTP $CACHE"
  fi

  # 5-1. 단일 예측 (--full 옵션일 때만)
  if [[ $FULL == true ]]; then
    echo -e "  ${YELLOW}⏳${RESET} 단일 예측 (AAPL) 실행 중... 최대 2분 소요"
    PRED=$(curl -s --max-time 130 -X POST "$ML/api/v1/predict" \
      -H "Content-Type: application/json" \
      -d '{"ticker":"AAPL","period_days":400,"include_sentiment":false,"force_lstm":false}' \
      2>/dev/null || echo "{}")
    if echo "$PRED" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('signal') in ('BUY','HOLD','SELL')" 2>/dev/null; then
      SIG=$(echo "$PRED" | python3 -c "import sys,json; print(json.load(sys.stdin)['signal'])" 2>/dev/null)
      pass "단일 예측 (AAPL) → signal: $SIG"
    else
      fail "단일 예측 실패: $(echo "$PRED" | head -c 200)"
    fi
  else
    skip "단일 예측 (--full 옵션으로 활성화)"
  fi
fi

# ── 6. WebSocket 테스트 ────────────────────────────────
section "4. WebSocket 테스트"

WS_SCRIPT="$(dirname "$0")/../scripts/test_ws.py"
WS_VENV="/tmp/stockai-ws-venv"
WS_PYTHON="python3"

# websockets 없으면 임시 venv에 자동 설치
if ! python3 -c "import websockets" 2>/dev/null; then
  echo -e "  ${YELLOW}⏳${RESET} websockets 없음 → 임시 venv 준비 중..."
  if [[ ! -x "$WS_VENV/bin/python3" ]]; then
    python3 -m venv "$WS_VENV" 2>/dev/null
    "$WS_VENV/bin/pip" install websockets -q 2>/dev/null
  fi
  if "$WS_VENV/bin/python3" -c "import websockets" 2>/dev/null; then
    WS_PYTHON="$WS_VENV/bin/python3"
  fi
fi

if [[ -z ${SCAN_JOB_ID:-} ]]; then
  skip "WebSocket 테스트 (스캔 job_id 없음 — ML 테스트가 건너뛰어졌거나 실패)"
elif ! $WS_PYTHON -c "import websockets" 2>/dev/null; then
  skip "WebSocket 테스트 (websockets venv 설치 실패)"
else
  WS_OUT=$($WS_PYTHON "$WS_SCRIPT" "$SCAN_JOB_ID" 2>&1 || echo "ERROR")
  if echo "$WS_OUT" | grep -q "completed\|failed\|running\|queued"; then
    pass "WebSocket 연결 및 메시지 수신"
  else
    fail "WebSocket 테스트 실패: $WS_OUT"
  fi
fi

# ── 8. 서비스 간 연동 ──────────────────────────────────
section "5. 서비스 간 연동 테스트"

# 시나리오 1: 백엔드 예측 이력 조회 (GET /api/v1/predictions/{ticker})
# --skip-ml 시 DB에 예측 데이터가 없으므로 ML을 호출하게 되고 503이 정상 응답임
if [[ -n ${TOKEN:-} ]]; then
  PROXY=$(curl -s -o /dev/null -w "%{http_code}" --max-time 150 \
    -X GET "$BACKEND/api/v1/predictions/MSFT" \
    -H "Authorization: Bearer $TOKEN" 2>/dev/null || echo "000")
  if [[ $PROXY == "200" || $PROXY == "404" ]]; then
    pass "시나리오 1: Backend 예측 이력 조회 → HTTP $PROXY"
  elif [[ $PROXY == "503" && $SKIP_ML == true ]]; then
    pass "시나리오 1: Backend 예측 이력 조회 → HTTP 503 (ML 없음, 정상)"
  else
    fail "시나리오 1: Backend 예측 이력 조회 → HTTP $PROXY"
  fi
fi

# Redis Celery 큐 확인
QLEN=$(docker compose -f "$(dirname "$0")/../docker-compose.yml" exec -T redis \
  redis-cli -n 2 LLEN celery 2>/dev/null | tr -d '[:space:]' || echo "err")
if [[ $QLEN =~ ^[0-9]+$ ]]; then
  pass "Redis Celery 큐 접근 가능 (대기 작업: $QLEN)"
else
  fail "Redis Celery 큐 확인 실패"
fi

# ── 결과 요약 ─────────────────────────────────────────
echo -e "\n${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
TOTAL=$((PASS + FAIL))
if [[ $FAIL -eq 0 ]]; then
  echo -e "${GREEN}${BOLD}결과: $PASS/$TOTAL 통과 ✓${RESET}"
else
  echo -e "${RED}${BOLD}결과: $PASS/$TOTAL 통과, $FAIL 실패${RESET}"
fi
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"

[[ $FAIL -eq 0 ]]