#!/usr/bin/env bash
# EC2 배포 스크립트 — GitHub Actions CD에서 SSH로 호출됨
set -euo pipefail

APP_DIR="/app"

echo "[deploy] Pulling latest code..."
cd "$APP_DIR"
git pull origin main

echo "[deploy] Building and restarting services..."
/usr/local/bin/docker-compose -f docker-compose.yml -f docker-compose.prod.yml \
  up --build -d --remove-orphans

echo "[deploy] Waiting for health checks..."
sleep 10
curl -sf http://localhost/api/health && echo " backend OK"
curl -sf http://localhost/ml/health && echo " ml OK"

echo "[deploy] Done."
