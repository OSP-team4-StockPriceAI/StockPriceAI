"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { SkeletonLoader } from "@/components/ui/SkeletonLoader";
import { watchlistApi, stocksApi, predictionsApi } from "@/lib/api";
import { Badge, signalToBadgeVariant } from "@/components/ui/badge";
import type { WatchlistItemOut } from "@/types/api";

function WatchlistRow({ item, onRemove }: { item: WatchlistItemOut; onRemove: (ticker: string) => void }) {
  const [isExpanded, setIsExpanded] = useState(false);

  const { data: stock } = useQuery({
    queryKey: ["stock", item.ticker],
    queryFn: () => stocksApi.get(item.ticker),
  });

  const { data: predictions } = useQuery({
    queryKey: ["predictions-history", item.ticker],
    queryFn: () => predictionsApi.get(item.ticker),
  });

  const prediction = predictions && predictions.length > 0 ? predictions[0] : undefined;

  const memoText = item.memo ?? "";
  const isLongMemo = memoText.length > 30;

  return (
    <tr className="border-b border-[#2d3748]/50 hover:bg-[#2d3748]/30 transition-colors">
      <td className="px-4 pt-4 pb-3 align-top font-mono font-bold text-[#e2e8f0]">
        {item.ticker}
      </td>
      <td className="px-4 pt-4 pb-3 align-top text-sm text-[#a0aec0]">
        {stock?.name ?? "—"}
      </td>
      <td className="px-4 pt-4 pb-3 align-top text-right text-[#e2e8f0]">
        {stock?.current_price ? `$${stock.current_price.toFixed(2)}` : "—"}
      </td>
      <td className="px-4 pt-4 pb-3 align-top text-right">
        {prediction ? (
          <Badge variant={signalToBadgeVariant(prediction.signal)}>
            {prediction.signal}
          </Badge>
        ) : "—"}
      </td>
      <td className="px-4 pt-4 pb-3 align-top text-right text-[#718096] hidden sm:table-cell">
        {prediction ? `${(prediction.up_prob * 100).toFixed(1)}%` : "—"}
      </td>
      <td className="px-4 pt-4 pb-3 align-top text-[#718096] text-sm hidden md:table-cell max-w-xs break-words whitespace-normal">
        {isLongMemo ? (
          <div>
            <span>{isExpanded ? memoText : `${memoText.slice(0, 30)}...`}</span>
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="ml-2 text-xs font-semibold text-blue-400 hover:text-blue-300 hover:underline focus:outline-none whitespace-nowrap"
            >
              {isExpanded ? "접기" : "더보기"}
            </button>
          </div>
        ) : (
          memoText
        )}
      </td>
      <td className="px-4 pt-4 pb-3 align-top">
        <button
          onClick={() => onRemove(item.ticker)}
          className="text-xs text-red-400 hover:text-red-300 transition-all hover-spring px-2 py-1 rounded hover:bg-red-500/10 active:scale-95"
        >
          삭제
        </button>
      </td>
    </tr>
  );
}

export function WatchlistTable() {
  const [ticker, setTicker] = useState("");
  const [memo, setMemo] = useState("");
  const queryClient = useQueryClient();

  const { data: items, isLoading } = useQuery({
    queryKey: ["watchlist"],
    queryFn: () => watchlistApi.list(),
  });

  const addMutation = useMutation({
    mutationFn: () =>
      watchlistApi.add({ ticker: ticker.toUpperCase(), memo: memo || undefined }),
    onSuccess: () => {
      setTicker("");
      setMemo("");
      queryClient.invalidateQueries({ queryKey: ["watchlist"] });
    },
  });

  const removeMutation = useMutation({
    mutationFn: (t: string) => watchlistApi.remove(t),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["watchlist"] }),
  });

  if (isLoading) return <SkeletonLoader type="table" count={5} />;

  return (
    <div className="space-y-4">
      {/* 추가 폼 */}
      <Card>
        <CardHeader>
          <CardTitle>종목 추가</CardTitle>
        </CardHeader>
        <form
          onSubmit={(e) => { e.preventDefault(); addMutation.mutate(); }}
          className="flex flex-wrap gap-2"
        >
          <Input
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            placeholder="티커 (예: AAPL)"
            className="w-36"
            required
          />
          <Input
            value={memo}
            onChange={(e) => setMemo(e.target.value)}
            placeholder="메모 (선택)"
            className="flex-1"
          />
          <Button type="submit" loading={addMutation.isPending} disabled={!ticker}>
            추가
          </Button>
        </form>
        {addMutation.isError && (
          <p className="mt-2 text-xs text-red-400">{(addMutation.error as Error).message}</p>
        )}
      </Card>

      {/* 목록 */}
      <Card className="overflow-x-auto p-0">
        <CardHeader className="p-4 pb-0">
          <CardTitle>관심종목 ({items?.length ?? 0}개)</CardTitle>
        </CardHeader>
        {!items || items.length === 0 ? (
          <div className="py-12 text-center text-[#718096]">
            관심종목이 없습니다. 위에서 종목을 추가해보세요.
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#2d3748]">
                <th className="px-4 py-3 text-left text-[#718096]">티커</th>
                <th className="px-4 py-3 text-left text-[#718096]">종목명</th>
                <th className="px-4 py-3 text-right text-[#718096]">현재가</th>
                <th className="px-4 py-3 text-right text-[#718096]">신호</th>
                <th className="px-4 py-3 text-right text-[#718096] hidden sm:table-cell">상승 확률</th>
                <th className="px-4 py-3 text-left text-[#718096] hidden md:table-cell">메모</th>
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <WatchlistRow
                  key={item.id}
                  item={item}
                  onRemove={(t) => removeMutation.mutate(t)}
                />
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
}
