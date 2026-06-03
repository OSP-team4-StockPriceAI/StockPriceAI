from pydantic import BaseModel


class Candle(BaseModel):
    Date: str | None = None
    Open: float | None = None
    High: float | None = None
    Low: float | None = None
    Close: float | None = None
    Volume: float | None = None


class StockInfo(BaseModel):
    ticker: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    current_price: float | None = None
    currency: str | None = None
    history: list[Candle] | None = None

