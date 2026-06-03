import type {
  LoginRequest,
  PredictRequest,
  PredictResponse,
  PredictionOut,
  RegisterRequest,
  ScanJobCreate,
  ScanJobOut,
  ScanResultOut,
  SentimentResponse,
  StockInfo,
  TechnicalResponse,
  TokenResponse,
  UserOut,
  WatchlistItemCreate,
  WatchlistItemOut,
} from "@/types/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const ML_BASE = process.env.NEXT_PUBLIC_ML_URL ?? "http://localhost:8001";

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("access_token");
}

async function request<T>(
  url: string,
  options: RequestInit = {},
  auth = true,
): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };

  if (auth) {
    const token = getToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(url, { ...options, headers });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Request failed");
  }

  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export const authApi = {
  register: (data: RegisterRequest) =>
    request<UserOut>(`${API_BASE}/api/v1/auth/register`, {
      method: "POST",
      body: JSON.stringify(data),
    }, false),

  login: (data: LoginRequest) =>
    request<TokenResponse>(`${API_BASE}/api/v1/auth/login`, {
      method: "POST",
      body: JSON.stringify(data),
    }, false),

  refresh: (refreshToken: string) =>
    request<TokenResponse>(`${API_BASE}/api/v1/auth/refresh`, {
      method: "POST",
      body: JSON.stringify({ refresh_token: refreshToken }),
    }, false),

  me: () => request<UserOut>(`${API_BASE}/api/v1/auth/me`),
};

// ── Stocks ────────────────────────────────────────────────────────────────────

export const stocksApi = {
  get: (ticker: string) =>
    request<StockInfo>(`${API_BASE}/api/v1/stocks/${ticker.toUpperCase()}`),
};

// ── Watchlist ─────────────────────────────────────────────────────────────────

export const watchlistApi = {
  list: () => request<WatchlistItemOut[]>(`${API_BASE}/api/v1/watchlist`),

  add: (data: WatchlistItemCreate) =>
    request<WatchlistItemOut>(`${API_BASE}/api/v1/watchlist`, {
      method: "POST",
      body: JSON.stringify(data),
    }),

  remove: (ticker: string) =>
    request<void>(`${API_BASE}/api/v1/watchlist/${ticker.toUpperCase()}`, {
      method: "DELETE",
    }),
};

// ── Predictions ───────────────────────────────────────────────────────────────

export const predictionsApi = {
  get: (ticker: string) =>
    request<PredictionOut[]>(`${API_BASE}/api/v1/predictions/${ticker.toUpperCase()}`),
};

// ── Scanner ───────────────────────────────────────────────────────────────────

export const scannerApi = {
  createJob: (data: ScanJobCreate) =>
    request<ScanJobOut>(`${API_BASE}/api/v1/scanner/jobs`, {
      method: "POST",
      body: JSON.stringify(data),
    }),

  getJob: (jobId: string) =>
    request<ScanJobOut>(`${API_BASE}/api/v1/scanner/jobs/${jobId}`),

  listJobs: () =>
    request<ScanJobOut[]>(`${API_BASE}/api/v1/scanner/jobs`),

  getResults: (jobId: string) =>
    request<ScanResultOut[]>(`${API_BASE}/api/v1/scanner/jobs/${jobId}/results`),
};

// ── ML — Technical ────────────────────────────────────────────────────────────

export const technicalApi = {
  get: (ticker: string, periodDays = 365) =>
    request<TechnicalResponse>(
      `${ML_BASE}/api/v1/technical/${ticker.toUpperCase()}?period_days=${periodDays}`,
      {},
      false,
    ),
};

// ── ML — Sentiment ────────────────────────────────────────────────────────────

export const sentimentApi = {
  get: (ticker: string) =>
    request<SentimentResponse>(
      `${ML_BASE}/api/v1/sentiment/${ticker.toUpperCase()}`,
      {},
      false,
    ),
};

// ── ML — Predict ──────────────────────────────────────────────────────────────

export const mlPredictApi = {
  predict: (data: PredictRequest) =>
    request<PredictResponse>(`${ML_BASE}/api/v1/predict`, {
      method: "POST",
      body: JSON.stringify(data),
    }, false),
};
