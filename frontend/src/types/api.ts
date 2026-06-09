// ── Auth ──────────────────────────────────────────────────────────────────────

export interface RegisterRequest {
  email: string;
  password: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface UserOut {
  id: string;
  email: string;
}

// ── Stock ─────────────────────────────────────────────────────────────────────

export interface StockInfo {
  ticker: string;
  name?: string;
  sector?: string;
  industry?: string;
  market_cap?: number;
  current_price?: number;
  currency?: string;
  history?: {
    Date: string;
    Open: number;
    High: number;
    Low: number;
    Close: number;
    Volume: number;
  }[];
}

// ── Watchlist ─────────────────────────────────────────────────────────────────

export interface WatchlistItemCreate {
  ticker: string;
  memo?: string;
}

export interface WatchlistItemOut {
  id: string;
  ticker: string;
  memo?: string;
  added_at: string;
}

// ── Prediction ────────────────────────────────────────────────────────────────

export interface PredictionOut {
  id: string;
  ticker: string;
  signal: string;
  up_prob: number;
  model_type?: string;
  complexity?: number;
  xgb_weight?: number;
  lstm_weight?: number;
  created_at: string;
}

// ── Scanner ───────────────────────────────────────────────────────────────────

export interface ScanJobCreate {
  sector?: string;
}

export interface ScanJobOut {
  id: string;
  status: string;
  total?: number;
  processed: number;
  sector?: string;
  started_at?: string;
  finished_at?: string;
  created_at: string;
}

export interface ScanResultOut {
  id: string;
  job_id: string;
  ticker: string;
  composite_score?: number;
  up_prob?: number;
  signal?: string;
  sector?: string;
  est_upside?: number;
  cached_at: string;
}

// ── ML — Technical ────────────────────────────────────────────────────────────

export interface SignalItem {
  action: string;
  description: string;
  color: string;
}

export interface TechnicalResponse {
  ticker: string;
  period_days: number;
  data_points: number;
  signals: Record<string, SignalItem>;
  support_resistance: Record<string, number>;
  latest_indicators: Record<string, number | null>;
  ma_trend: string;
  overall_signal: string;
}

// ── ML — Sentiment ────────────────────────────────────────────────────────────

export interface NewsItem {
  title: string;
  publisher: string;
  hours_ago: number;
  source: string;
  compound: number;
  label: string;
  relevance: number;
  relevance_tier: string;
  news_type: string;
  impact_score: number;
  macro_theme?: string;
}

export interface SentimentResponse {
  ticker: string;
  signal: string;
  avg_sentiment: number;
  time_weighted_avg: number;
  raw_avg: number;
  impact_score_avg: number;
  positive_pct: number;
  negative_pct: number;
  neutral_pct: number;
  news_count: number;
  direct_news_count: number;
  surprise_count: number;
  structural_count: number;
  macro_themes: string[];
  model: string;
  sources: string[];
  news: NewsItem[];
}

// ── ML — Predict ──────────────────────────────────────────────────────────────

export interface PredictRequest {
  ticker: string;
  period_days?: number;
}

export interface PredictResponse {
  ticker: string;
  signal: string;
  up_prob: number;
  model_type: string;
  complexity?: number;
  xgb_weight?: number;
  lstm_weight?: number;
}

// ── WebSocket ─────────────────────────────────────────────────────────────────

export interface ScanProgressMessage {
  type: "progress" | "complete" | "error";
  job_id: string;
  processed?: number;
  total?: number;
  ticker?: string;
  message?: string;
}

// ── Common ────────────────────────────────────────────────────────────────────

export interface ApiError {
  detail: string;
}
