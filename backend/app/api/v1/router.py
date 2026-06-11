from fastapi import APIRouter

from app.api.v1.endpoints import auth, predictions, scanner, sentiment, stocks, watchlist, websocket

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth.router)
api_router.include_router(watchlist.router)
api_router.include_router(stocks.router)
api_router.include_router(predictions.router)
api_router.include_router(scanner.router)
api_router.include_router(sentiment.router)

# WebSocket은 prefix 없이 루트에 등록
ws_router = APIRouter()
ws_router.include_router(websocket.router)