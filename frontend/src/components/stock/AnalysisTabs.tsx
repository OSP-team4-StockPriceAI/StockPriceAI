"use client";

import { useQuery } from "@tanstack/react-query";
import { Tabs, TabPanel } from "@/components/ui/tabs";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge, signalToBadgeVariant } from "@/components/ui/badge";
import { SkeletonLoader } from "@/components/ui/SkeletonLoader";
import { TechnicalIndicatorsChart } from "@/components/charts/TechnicalChart";
import { technicalApi, sentimentApi, predictionsApi } from "@/lib/api";
import { useUIStore } from "@/store/ui";
import { useAuthStore } from "@/store/auth";

const TABS = [
  { id: "overview", label: "개요" },
  { id: "technical", label: "기술적 지표" },
  { id: "sentiment", label: "감성 분석" },
  { id: "prediction", label: "AI 예측" },
  { id: "signals", label: "매매 신호" },
  { id: "support", label: "지지/저항" },
  { id: "indicators", label: "지표 상세" },
];

interface AnalysisTabsProps {
  ticker: string;
}

export function AnalysisTabs({ ticker }: AnalysisTabsProps) {
  const { activeTab, setActiveTab } = useUIStore();
  const { isAuthenticated } = useAuthStore();

  const { data: technical, isLoading: techLoading } = useQuery({
    queryKey: ["technical", ticker],
    queryFn: () => technicalApi.get(ticker),
    enabled: !!ticker,
  });

  const { data: sentiment, isLoading: sentLoading } = useQuery({
    queryKey: ["sentiment", ticker],
    queryFn: () => sentimentApi.get(ticker),
    enabled: !!ticker && activeTab === "sentiment",
  });

  const { data: predictions, isLoading: predLoading } = useQuery({
    queryKey: ["predictions-history", ticker],
    queryFn: () => predictionsApi.get(ticker),
    enabled: !!ticker && activeTab === "prediction" && isAuthenticated,
  });

  const prediction = predictions && predictions.length > 0 ? predictions[0] : undefined;

  return (
    <Tabs tabs={TABS} activeTab={activeTab} onChange={setActiveTab}>
      {/* 개요 탭 */}
      <TabPanel id="overview" activeTab={activeTab}>
        {techLoading ? (
          <SkeletonLoader type="tab-content" />
        ) : technical ? (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Card>
                <div className="text-xs text-[#718096]">전반적 신호</div>
                <Badge
                  variant={signalToBadgeVariant(technical.overall_signal)}
                  className="mt-2"
                >
                  {technical.overall_signal}
                </Badge>
              </Card>
              <Card>
                <div className="text-xs text-[#718096]">MA 추세</div>
                <div className="mt-2 text-sm font-medium capitalize text-[#e2e8f0]">
                  {technical.ma_trend}
                </div>
              </Card>
              <Card>
                <div className="text-xs text-[#718096]">분석 기간</div>
                <div className="mt-2 text-sm text-[#e2e8f0]">
                  {technical.period_days}일
                </div>
              </Card>
              <Card>
                <div className="text-xs text-[#718096]">데이터 포인트</div>
                <div className="mt-2 text-sm text-[#e2e8f0]">
                  {technical.data_points.toLocaleString()}
                </div>
              </Card>
            </div>
            <TechnicalIndicatorsChart data={technical} />
          </div>
        ) : null}
      </TabPanel>

      {/* 기술적 지표 탭 */}
      <TabPanel id="technical" activeTab={activeTab}>
        {techLoading ? (
          <SkeletonLoader type="card-grid" count={8} />
        ) : technical ? (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {Object.entries(technical.latest_indicators)
              .filter(([, v]) => v !== null)
              .map(([key, value]) => (
                <Card key={key}>
                  <div className="text-xs text-[#718096] uppercase">{key.replace(/_/g, " ")}</div>
                  <div className="mt-1 text-lg font-semibold text-[#e2e8f0]">
                    {typeof value === "number" ? value.toFixed(4) : "—"}
                  </div>
                </Card>
              ))}
          </div>
        ) : null}
      </TabPanel>

      {/* 감성 분석 탭 */}
      <TabPanel id="sentiment" activeTab={activeTab}>
        {sentLoading ? (
          <SkeletonLoader type="tab-content" />
        ) : sentiment ? (
          <div className="space-y-6">
            {/* 상단 3개 핵심 요약 카드 */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <Card className="flex flex-col justify-between">
                <div>
                  <div className="text-xs text-[#718096]">전반적 감성 신호</div>
                  <Badge
                    variant={
                      sentiment.signal === "BULLISH"
                        ? "buy"
                        : sentiment.signal === "BEARISH"
                          ? "sell"
                          : "hold"
                    }
                    className="mt-2 text-sm px-3 py-1 font-bold"
                  >
                    {sentiment.signal}
                  </Badge>
                </div>
                <div className="text-[10px] text-[#718096] mt-2">
                  뉴스 임팩트 점수를 가중 반영한 최종 신호입니다.
                </div>
              </Card>
              
              <Card>
                <div className="text-xs text-[#718096]">평균 감성 점수 (VADER)</div>
                <div className="mt-2 text-3xl font-extrabold text-[#e2e8f0]">
                  {sentiment.avg_sentiment.toFixed(3)}
                </div>
                <div className="text-[10px] text-[#718096] mt-2">
                  범위: -1.0 (매우 부정) ~ +1.0 (매우 긍정)
                </div>
              </Card>

              <Card>
                <div className="text-xs text-[#718096]">수집 및 분석 뉴스 수</div>
                <div className="mt-2 text-3xl font-extrabold text-[#e2e8f0]">
                  {sentiment.news_count} <span className="text-sm font-normal text-[#718096]">건</span>
                </div>
                <div className="text-[10px] text-[#718096] mt-2">
                  직접 연관 뉴스: {sentiment.direct_news_count}건
                </div>
              </Card>
            </div>

            {/* 감성 분포 프로그레스 바 */}
            <Card className="p-4 bg-[#1a202c]/20">
              <div className="text-xs font-semibold text-[#a0aec0] mb-3">감성 뉴스 분포 비율</div>
              <div className="w-full flex h-4 rounded-full overflow-hidden bg-[#2d3748] border border-[#2d3748]">
                {sentiment.positive_pct > 0 && (
                  <div 
                    className="bg-emerald-500 h-full flex items-center justify-center text-[10px] font-bold text-white transition-all"
                    style={{ width: `${sentiment.positive_pct}%` }}
                    title={`긍정: ${sentiment.positive_pct}%`}
                  >
                    {sentiment.positive_pct >= 10 ? `${sentiment.positive_pct}%` : ""}
                  </div>
                )}
                {sentiment.neutral_pct > 0 && (
                  <div 
                    className="bg-[#4a5568] h-full flex items-center justify-center text-[10px] font-bold text-[#a0aec0] transition-all"
                    style={{ width: `${sentiment.neutral_pct}%` }}
                    title={`중립: ${sentiment.neutral_pct}%`}
                  >
                    {sentiment.neutral_pct >= 10 ? `${sentiment.neutral_pct}%` : ""}
                  </div>
                )}
                {sentiment.negative_pct > 0 && (
                  <div 
                    className="bg-red-500 h-full flex items-center justify-center text-[10px] font-bold text-white transition-all"
                    style={{ width: `${sentiment.negative_pct}%` }}
                    title={`부정: ${sentiment.negative_pct}%`}
                  >
                    {sentiment.negative_pct >= 10 ? `${sentiment.negative_pct}%` : ""}
                  </div>
                )}
              </div>
              <div className="flex justify-between text-xs mt-2 font-mono">
                <span className="text-emerald-400">● 긍정: {sentiment.positive_pct}%</span>
                <span className="text-[#a0aec0]">● 중립: {sentiment.neutral_pct}%</span>
                <span className="text-red-400">● 부정: {sentiment.negative_pct}%</span>
              </div>
            </Card>

            {/* 매크로 테마 & 지표 분석 */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <Card>
                <div className="text-xs text-[#718096] mb-2">분석된 주요 매크로 테마</div>
                {sentiment.macro_themes && sentiment.macro_themes.length > 0 ? (
                  <div className="flex flex-wrap gap-1.5 mt-1">
                    {sentiment.macro_themes.map((theme) => (
                      <Badge key={theme} variant="default" className="text-[10px] capitalize bg-[#2d3748]/50 border-[#4a5568]/40">
                        {theme}
                      </Badge>
                    ))}
                  </div>
                ) : (
                  <div className="text-sm text-[#718096] italic mt-1">특이 매크로 테마가 없습니다.</div>
                )}
              </Card>

              <Card className="grid grid-cols-2 gap-2 p-3">
                <div>
                  <div className="text-[10px] text-[#718096]">시간 가중 감성 평균</div>
                  <div className="text-sm font-bold text-[#e2e8f0] mt-0.5">{sentiment.time_weighted_avg.toFixed(3)}</div>
                </div>
                <div>
                  <div className="text-[10px] text-[#718096]">임팩트 가중 평균</div>
                  <div className="text-sm font-bold text-[#e2e8f0] mt-0.5">{sentiment.impact_score_avg.toFixed(3)}</div>
                </div>
                <div>
                  <div className="text-[10px] text-[#718096]">어닝/이벤트 어프라이즈</div>
                  <div className="text-sm font-bold text-emerald-400 mt-0.5">{sentiment.surprise_count}건</div>
                </div>
                <div>
                  <div className="text-[10px] text-[#718096]">구조적 변동(Structural)</div>
                  <div className="text-sm font-bold text-blue-400 mt-0.5">{sentiment.structural_count}건</div>
                </div>
              </Card>
            </div>

            {/* 실시간 뉴스 리스트 */}
            <Card className="overflow-x-auto p-0">
              <CardHeader className="p-4 pb-0">
                <CardTitle className="text-sm font-bold flex items-center justify-between">
                  <span>감성 분석된 실시간 뉴스 리스트</span>
                  <span className="text-xs font-normal text-[#718096]">최근 최대 30개 건</span>
                </CardTitle>
              </CardHeader>
              {!sentiment.news || sentiment.news.length === 0 ? (
                <div className="py-8 text-center text-[#718096] text-xs italic">
                  최근 90일 내 수집된 관련 뉴스가 없습니다.
                </div>
              ) : (
                <table className="w-full text-xs text-left">
                  <thead>
                    <tr className="border-b border-[#2d3748] bg-[#1a202c]/50">
                      <th className="px-4 py-2 text-[#718096] font-semibold">뉴스 제목</th>
                      <th className="px-4 py-2 text-[#718096] font-semibold text-center">감성</th>
                      <th className="px-4 py-2 text-[#718096] font-semibold text-right">임팩트</th>
                      <th className="px-4 py-2 text-[#718096] font-semibold">매크로</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sentiment.news.map((item, idx) => (
                      <tr key={idx} className="border-b border-[#2d3748]/30 hover:bg-[#2d3748]/15 transition-colors">
                        <td className="px-4 py-2 max-w-sm sm:max-w-md">
                          <div className="font-medium text-[#e2e8f0] line-clamp-1">{item.title}</div>
                          <div className="flex items-center gap-1.5 text-[10px] text-[#718096] mt-0.5">
                            <span className="font-semibold text-[#a0aec0]">{item.publisher}</span>
                            <span>•</span>
                            <span>{item.hours_ago < 1 ? "방금 전" : `${item.hours_ago.toFixed(1)}시간 전`}</span>
                          </div>
                        </td>
                        <td className="px-4 py-2 text-center">
                          <Badge
                            variant={
                              item.label === "BULLISH"
                                ? "buy"
                                : item.label === "BEARISH"
                                  ? "sell"
                                  : "hold"
                            }
                            className="text-[9px] px-1.5 py-0.2"
                          >
                            {item.label}
                          </Badge>
                        </td>
                        <td className="px-4 py-2 text-right font-mono font-bold">
                          <span className={item.impact_score > 0 ? "text-emerald-400" : item.impact_score < 0 ? "text-red-400" : "text-[#718096]"}>
                            {item.impact_score > 0 ? "+" : ""}{item.impact_score.toFixed(2)}
                          </span>
                        </td>
                        <td className="px-4 py-2">
                          {item.macro_theme ? (
                            <span className="text-[10px] font-mono text-blue-400/90 capitalize">{item.macro_theme}</span>
                          ) : (
                            <span className="text-[10px] text-[#4a5568]">—</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Card>
          </div>
        ) : (
          <div className="text-[#718096] text-sm">감성 분석 탭을 선택하면 로드됩니다.</div>
        )}
      </TabPanel>

      {/* AI 예측 탭 */}
      <TabPanel id="prediction" activeTab={activeTab}>
        {!isAuthenticated ? (
          <div className="flex flex-col items-center justify-center py-12 text-center border border-dashed border-[#2d3748] rounded-lg p-6 bg-[#1a202c]/30">
            <div className="text-4xl mb-3">🔒</div>
            <h3 className="text-md font-medium text-[#e2e8f0]">AI 예측은 회원 전용 기능입니다</h3>
            <p className="mt-1 text-sm text-[#718096]">로그인하시면 AI 분석과 예측 상승확률을 확인하실 수 있습니다.</p>
          </div>
        ) : predLoading ? (
          <SkeletonLoader type="card-grid" count={3} />
        ) : prediction ? (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              <Card>
                <div className="text-xs text-[#718096]">예측 신호</div>
                <Badge
                  variant={signalToBadgeVariant(prediction.signal)}
                  className="mt-2"
                >
                  {prediction.signal}
                </Badge>
              </Card>
              <Card>
                <div className="text-xs text-[#718096]">상승 확률</div>
                <div className="mt-2 text-2xl font-bold text-[#e2e8f0]">
                  {(prediction.up_prob * 100).toFixed(1)}%
                </div>
                <div className="w-full bg-[#2d3748] rounded-full h-2 mt-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full"
                    style={{ width: `${prediction.up_prob * 100}%` }}
                  />
                </div>
              </Card>
              <Card>
                <div className="text-xs text-[#718096]">모델 타입</div>
                <div className="mt-2 text-sm font-medium text-[#e2e8f0] capitalize">
                  {prediction.model_type}
                </div>
              </Card>
              {prediction.xgb_weight !== undefined && (
                <Card>
                  <div className="text-xs text-[#718096]">XGBoost 가중치</div>
                  <div className="mt-2 text-lg font-bold text-[#e2e8f0]">
                    {(prediction.xgb_weight * 100).toFixed(1)}%
                  </div>
                </Card>
              )}
              {prediction.lstm_weight !== undefined && (
                <Card>
                  <div className="text-xs text-[#718096]">LSTM 가중치</div>
                  <div className="mt-2 text-lg font-bold text-[#e2e8f0]">
                    {(prediction.lstm_weight * 100).toFixed(1)}%
                  </div>
                </Card>
              )}
              {prediction.complexity !== undefined && (
                <Card>
                  <div className="text-xs text-[#718096]">복잡도</div>
                  <div className="mt-2 text-lg font-bold text-[#e2e8f0]">
                    {prediction.complexity.toFixed(3)}
                  </div>
                </Card>
              )}
            </div>
          </div>
        ) : (
          <div className="text-[#718096] text-sm">AI 예측 탭을 선택하면 분석이 실행됩니다.</div>
        )}
      </TabPanel>

      {/* 매매 신호 탭 */}
      <TabPanel id="signals" activeTab={activeTab}>
        {techLoading ? (
          <SkeletonLoader type="card-grid" count={4} />
        ) : technical ? (
          <div className="grid gap-3 sm:grid-cols-2">
            {Object.entries(technical.signals).map(([name, signal]) => (
              <Card key={name} className="flex items-start justify-between">
                <div>
                  <div className="text-sm font-medium text-[#e2e8f0] capitalize">
                    {name.replace(/_/g, " ")}
                  </div>
                  <div className="mt-1 text-xs text-[#718096]">{signal.description}</div>
                </div>
                <Badge variant={signalToBadgeVariant(signal.action)}>
                  {signal.action}
                </Badge>
              </Card>
            ))}
          </div>
        ) : null}
      </TabPanel>

      {/* 지지/저항 탭 */}
      <TabPanel id="support" activeTab={activeTab}>
        {techLoading ? (
          <SkeletonLoader type="card-grid" count={4} />
        ) : technical ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {Object.entries(technical.support_resistance).map(([key, value]) => (
              <Card key={key}>
                <div className="text-xs text-[#718096] capitalize">
                  {key.replace(/_/g, " ")}
                </div>
                <div className="mt-1 text-lg font-bold text-[#e2e8f0]">
                  ${value.toFixed(2)}
                </div>
              </Card>
            ))}
          </div>
        ) : null}
      </TabPanel>

      {/* 지표 상세 탭 */}
      <TabPanel id="indicators" activeTab={activeTab}>
        {techLoading ? (
          <SkeletonLoader type="table" count={10} />
        ) : technical ? (
          <Card className="overflow-x-auto">
            <CardHeader>
              <CardTitle>최신 지표 값</CardTitle>
            </CardHeader>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[#2d3748]">
                  <th className="py-2 text-left text-[#718096]">지표</th>
                  <th className="py-2 text-right text-[#718096]">값</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(technical.latest_indicators).map(([key, value]) => (
                  <tr key={key} className="border-b border-[#2d3748]/50">
                    <td className="py-2 text-[#a0aec0] uppercase text-xs">{key}</td>
                    <td className="py-2 text-right font-mono text-[#e2e8f0]">
                      {value !== null ? value.toFixed(4) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        ) : null}
      </TabPanel>
    </Tabs>
  );
}
