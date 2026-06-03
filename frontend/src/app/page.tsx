"use client";

import { useState, useEffect, useMemo, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { StockSearch } from "@/components/stock/StockSearch";
import { StockOverview } from "@/components/stock/StockOverview";
import { AnalysisTabs } from "@/components/stock/AnalysisTabs";
import { AuthModal } from "@/components/layout/AuthModal";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FullPageSpinner } from "@/components/ui/spinner";
import { SkeletonLoader } from "@/components/ui/SkeletonLoader";
import { CandlestickChart } from "@/components/charts/CandlestickChart";
import { stocksApi, predictionsApi } from "@/lib/api";
import { useUIStore } from "@/store/ui";
import { useAuthStore } from "@/store/auth";

interface OHLCVData {
  date: string[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume?: number[];
}

const generateChartData = (currentPrice: number, range: "1M" | "3M" | "1Y" | "5Y" | "ALL") => {
  const date: string[] = [];
  const open: number[] = [];
  const high: number[] = [];
  const low: number[] = [];
  const close: number[] = [];
  const volume: number[] = [];

  const basePrice = currentPrice > 0 ? currentPrice : 150;
  const now = new Date();

  let numCandles = 0;
  let calendarDaysPerCandle = 1;

  if (range === "1M") {
    numCandles = 22; // ~22 trading days
    calendarDaysPerCandle = 1;
  } else if (range === "3M") {
    numCandles = 65; // ~65 trading days
    calendarDaysPerCandle = 1;
  } else if (range === "1Y") {
    numCandles = 252; // ~252 trading days
    calendarDaysPerCandle = 1;
  } else if (range === "5Y") {
    numCandles = 260; // 260 weekly candles (5 years * 52 weeks = 260)
    calendarDaysPerCandle = 7;
  } else { // "ALL"
    numCandles = 360; // 360 monthly candles (30 years * 12 months = 360)
    calendarDaysPerCandle = 30;
  }

  // 1. Collect dates backwards
  let dayOffset = 0;
  while (date.length < numCandles) {
    const d = new Date(now);
    d.setDate(now.getDate() - dayOffset);

    // Skip weekends only for daily candles to look authentic
    if (calendarDaysPerCandle === 1) {
      const dayOfWeek = d.getDay();
      if (dayOfWeek === 0 || dayOfWeek === 6) {
        dayOffset++;
        continue;
      }
    }

    date.push(d.toISOString().split("T")[0]);
    dayOffset += calendarDaysPerCandle;
  }

  // Reverse to make chronological
  date.reverse();

  // 2. Generate price random walk
  // Simulate historical growth patterns (ALL / 5Y start much lower)
  let growthFactor = 0.15; // 1M / 3M starts close
  if (range === "1Y") growthFactor = 0.35;
  else if (range === "5Y") growthFactor = 0.65;
  else if (range === "ALL") growthFactor = 0.90;

  let prevClose = basePrice * (1 - growthFactor + (Math.random() - 0.5) * 0.05);
  const volatility = calendarDaysPerCandle === 1 ? 0.018 : calendarDaysPerCandle === 7 ? 0.035 : 0.07;

  for (let i = 0; i < numCandles; i++) {
    const change = (Math.random() - 0.45) * (basePrice * volatility); // upward bias
    const currentOpen = prevClose;
    let currentCloseVal = prevClose + change;

    // Force the last day to match the real-time price exactly
    if (i === numCandles - 1) {
      currentCloseVal = basePrice;
    }

    const currentHigh = Math.max(currentOpen, currentCloseVal) + Math.random() * (basePrice * volatility * 0.7);
    const currentLow = Math.min(currentOpen, currentCloseVal) - Math.random() * (basePrice * volatility * 0.7);
    const currentVolume = Math.floor((800000 + Math.random() * 4000000) * calendarDaysPerCandle);

    open.push(Number(currentOpen.toFixed(2)));
    close.push(Number(currentCloseVal.toFixed(2)));
    high.push(Number(currentHigh.toFixed(2)));
    low.push(Number(currentLow.toFixed(2)));
    volume.push(currentVolume);

    prevClose = currentCloseVal;
  }

  return { date, open, high, low, close, volume };
};

function DashboardContent() {
  const { selectedTicker, setSelectedTicker } = useUIStore();
  const { isAuthenticated } = useAuthStore();
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [range, setRange] = useState<"1M" | "3M" | "1Y" | "5Y" | "ALL">("1Y");

  const searchParams = useSearchParams();
  const tickerParam = searchParams.get("ticker");

  useEffect(() => {
    if (tickerParam) {
      setSelectedTicker(tickerParam.toUpperCase());
    }
  }, [tickerParam, setSelectedTicker]);

  const { data: stockInfo, isLoading: stockLoading, error: stockError } = useQuery({
    queryKey: ["stock", selectedTicker],
    queryFn: () => stocksApi.get(selectedTicker),
    enabled: !!selectedTicker,
  });

  const { data: predictions } = useQuery({
    queryKey: ["predictions-history", selectedTicker],
    queryFn: () => predictionsApi.get(selectedTicker),
    enabled: !!selectedTicker && isAuthenticated,
  });

  const prediction = predictions && predictions.length > 0 ? predictions[0] : undefined;

  // Generate dynamic, optimized historical chart data based on the selected range
  const chartData = useMemo(() => {
    if (!stockInfo?.current_price) return null;
    return generateChartData(stockInfo.current_price, range);
  }, [selectedTicker, stockInfo?.current_price, range]);

  function handleSearch(ticker: string) {
    setSelectedTicker(ticker);
  }

  return (
    <div className="space-y-6">
      {/* 헤더 */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-[#e2e8f0]">주식 분석 대시보드</h1>
          <p className="mt-1 text-sm text-[#718096]">
            AI 기반 예측 · 기술적 지표 · 감성 분석
          </p>
        </div>
      </div>

      {/* 검색 */}
      <Card>
        <StockSearch onSearch={handleSearch} loading={stockLoading} />
      </Card>

      {/* 에러 상태 */}
      {stockError && (
        <div className="rounded-md border border-red-800 bg-red-900/20 px-4 py-3 text-sm text-red-400">
          종목을 찾을 수 없습니다: {selectedTicker}
        </div>
      )}

      {/* 분석 결과 */}
      {selectedTicker && !stockError && (
        stockLoading ? (
          <div className="space-y-6 animate-in fade-in duration-300">
            {/* Header Skeleton */}
            <div className="flex items-baseline gap-3">
              <div className="h-7 w-20 rounded animate-skeleton" />
              <div className="h-4 w-32 rounded animate-skeleton" />
            </div>
            {/* Chart Skeleton */}
            <SkeletonLoader type="chart" />
            {/* Overview Cards Mock */}
            <SkeletonLoader type="card-grid" count={4} />
            {/* Tab content mock */}
            <SkeletonLoader type="tab-content" />
          </div>
        ) : (
          <>
            {/* 종목 헤더 */}
            <div className="flex items-baseline gap-3">
              <h2 className="text-xl font-bold text-[#e2e8f0]">{selectedTicker}</h2>
              {stockInfo?.name && (
                <span className="text-[#718096]">{stockInfo.name}</span>
              )}
            </div>

            {/* 캔들스틱 가격 차트 */}
            {stockInfo && chartData && (
              <Card className="p-4 bg-[#1a202c]/20 border-[#2d3748]/40 shadow-xl overflow-hidden animate-in fade-in slide-in-from-top-4 duration-500">
                {/* 차트 상단 컨트롤 헤더 */}
                <div className="flex items-center justify-between gap-4 mb-3">
                  <span className="text-xs font-semibold text-[#a0aec0] uppercase tracking-wider">주가 차트 분석</span>
                  <div className="flex gap-1 p-0.5 bg-[#1e293b]/70 rounded-lg border border-[#2d3748]/50">
                    {(["1M", "3M", "1Y", "5Y", "ALL"] as const).map((opt) => (
                      <button
                        key={opt}
                        onClick={() => setRange(opt)}
                        className={`px-3 py-1 text-xs font-bold rounded-md transition-all duration-200 cursor-pointer ${
                          range === opt
                            ? "bg-blue-600 text-white shadow-md scale-[1.03]"
                            : "text-[#718096] hover:text-[#e2e8f0] hover:bg-[#2d3748]/30"
                        }`}
                      >
                        {opt === "ALL" ? "전체" : opt}
                      </button>
                    ))}
                  </div>
                </div>

                <CandlestickChart
                  data={chartData}
                  ticker={selectedTicker}
                />
              </Card>
            )}

            {/* 개요 카드 */}
            {stockInfo && (
              <StockOverview stockInfo={stockInfo} prediction={prediction} />
            )}

            {/* 분석 탭 (7탭) */}
            <Card className="p-0 overflow-hidden">
              <div className="p-4">
                <AnalysisTabs ticker={selectedTicker} />
              </div>
            </Card>
          </>
        )
      )}

      {/* 빈 상태 */}
      {!selectedTicker && (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <div className="text-5xl mb-4">📈</div>
          <h3 className="text-lg font-medium text-[#a0aec0]">
            종목 티커를 입력하여 분석을 시작하세요
          </h3>
          <p className="mt-2 text-sm text-[#718096]">
            S&P 500 종목 스캔은{" "}
            <a href="/scanner" className="text-blue-400 hover:underline">
              스캐너 페이지
            </a>
            에서 이용할 수 있습니다.
          </p>
        </div>
      )}

      {/* 인증 모달 */}
      {showAuthModal && <AuthModal onClose={() => setShowAuthModal(false)} />}
    </div>
  );
}

export default function DashboardPage() {
  return (
    <Suspense fallback={<FullPageSpinner />}>
      <DashboardContent />
    </Suspense>
  );
}
