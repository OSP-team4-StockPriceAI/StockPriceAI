import { Card } from "@/components/ui/card";
import { Badge, signalToBadgeVariant } from "@/components/ui/badge";
import type { PredictionOut, StockInfo } from "@/types/api";

interface StockOverviewProps {
  stockInfo: StockInfo;
  prediction?: PredictionOut;
}

function formatMarketCap(cap?: number): string {
  if (!cap) return "—";
  if (cap >= 1e12) return `$${(cap / 1e12).toFixed(2)}T`;
  if (cap >= 1e9) return `$${(cap / 1e9).toFixed(2)}B`;
  if (cap >= 1e6) return `$${(cap / 1e6).toFixed(2)}M`;
  return `$${cap.toFixed(0)}`;
}

export function StockOverview({ stockInfo, prediction }: StockOverviewProps) {
  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
      <Card>
        <div className="text-xs text-[#718096] mb-1">현재 가격</div>
        <div className="text-2xl font-bold text-[#e2e8f0]">
          {stockInfo.current_price
            ? `$${stockInfo.current_price.toFixed(2)}`
            : "—"}
        </div>
        {stockInfo.currency && (
          <div className="text-xs text-[#718096] mt-1">{stockInfo.currency}</div>
        )}
      </Card>

      <Card>
        <div className="text-xs text-[#718096] mb-1">시가총액</div>
        <div className="text-xl font-bold text-[#e2e8f0]">
          {formatMarketCap(stockInfo.market_cap)}
        </div>
      </Card>

      <Card>
        <div className="text-xs text-[#718096] mb-1">섹터</div>
        <div className="text-sm font-medium text-[#e2e8f0] mt-1">
          {stockInfo.sector ?? "—"}
        </div>
        <div className="text-xs text-[#718096] mt-1">
          {stockInfo.industry ?? ""}
        </div>
      </Card>

      {prediction && (
        <Card>
          <div className="text-xs text-[#718096] mb-1">AI 예측 신호</div>
          <div className="flex items-center gap-2 mt-1">
            <Badge variant={signalToBadgeVariant(prediction.signal)}>
              {prediction.signal}
            </Badge>
          </div>
          <div className="text-xs text-[#718096] mt-2">
            상승 확률: {(prediction.up_prob * 100).toFixed(1)}%
          </div>
          <div className="w-full bg-[#2d3748] rounded-full h-1.5 mt-1">
            <div
              className="bg-blue-500 h-1.5 rounded-full"
              style={{ width: `${prediction.up_prob * 100}%` }}
            />
          </div>
        </Card>
      )}
    </div>
  );
}
