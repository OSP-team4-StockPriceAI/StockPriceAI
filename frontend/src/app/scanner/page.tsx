"use client";

import { useCallback, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { FullPageSpinner } from "@/components/ui/spinner";
import { ScannerProgress } from "@/components/scanner/ScannerProgress";
import { ScanResults } from "@/components/scanner/ScanResults";
import { AuthModal } from "@/components/layout/AuthModal";
import { scannerApi } from "@/lib/api";
import { useAuthStore } from "@/store/auth";
import type { ScanProgressMessage } from "@/types/api";

const SECTORS = [
  { value: "", label: "전체 S&P 500" },
  { value: "Technology", label: "기술" },
  { value: "Healthcare", label: "헬스케어" },
  { value: "Financials", label: "금융" },
  { value: "Consumer Discretionary", label: "소비재" },
  { value: "Industrials", label: "산업재" },
  { value: "Energy", label: "에너지" },
  { value: "Utilities", label: "유틸리티" },
  { value: "Real Estate", label: "부동산" },
  { value: "Materials", label: "소재" },
  { value: "Communication Services", label: "통신서비스" },
];

function statusToBadge(status: string) {
  switch (status) {
    case "running": return "warning";
    case "completed": return "success";
    case "failed": return "danger";
    default: return "default";
  }
}

export default function ScannerPage() {
  const { isAuthenticated } = useAuthStore();
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [selectedSector, setSelectedSector] = useState("");
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [lastMessage, setLastMessage] = useState<string>("");
  const queryClient = useQueryClient();

  const { data: jobs, isLoading: jobsLoading } = useQuery({
    queryKey: ["scanner-jobs"],
    queryFn: () => scannerApi.listJobs(),
    enabled: isAuthenticated,
    refetchInterval: 10_000,
  });

  const { data: activeJob, refetch: refetchJob } = useQuery({
    queryKey: ["scanner-job", activeJobId],
    queryFn: () => scannerApi.getJob(activeJobId!),
    enabled: !!activeJobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "running" || status === "pending" ? 3_000 : false;
    },
  });

  const { data: results } = useQuery({
    queryKey: ["scanner-results", activeJobId],
    queryFn: () => scannerApi.getResults(activeJobId!),
    enabled: !!activeJobId && activeJob?.status === "completed",
  });

  const createJobMutation = useMutation({
    mutationFn: () => scannerApi.createJob({ sector: selectedSector || undefined }),
    onSuccess: (job) => {
      setActiveJobId(job.id);
      queryClient.invalidateQueries({ queryKey: ["scanner-jobs"] });
    },
    onError: (err: Error) => {
      if (err.message.includes("401") || err.message.includes("인증")) {
        setShowAuthModal(true);
      }
    },
  });

  const handleProgress = useCallback((msg: ScanProgressMessage) => {
    if (msg.ticker) setLastMessage(`분석 중: ${msg.ticker}`);
    refetchJob();
  }, [refetchJob]);

  const handleComplete = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["scanner-results", activeJobId] });
    queryClient.invalidateQueries({ queryKey: ["scanner-jobs"] });
  }, [activeJobId, queryClient]);

  function handleStartScan() {
    if (!isAuthenticated) {
      setShowAuthModal(true);
      return;
    }
    setLastMessage("");
    createJobMutation.mutate();
  }

  return (
    <div className="flex flex-col lg:flex-row gap-6 relative items-start">
      <div className="flex-1 space-y-6 min-w-0">
        <div>
        <h1 className="text-2xl font-bold text-[#e2e8f0]">S&P 500 스캐너</h1>
        <p className="mt-1 text-sm text-[#718096]">
          AI 기반 배치 분석으로 매수 신호 종목을 스크리닝합니다
        </p>
      </div>

      {/* 스캔 설정 */}
      <Card id="scan-settings" className="space-y-4 scroll-mt-24">
        <h2 className="text-base font-semibold text-[#e2e8f0]">스캔 설정</h2>
        <div className="flex flex-wrap gap-3 items-end">
          <div className="flex-1 min-w-48">
            <label className="block text-sm text-[#a0aec0] mb-1">섹터 필터</label>
            <select
              value={selectedSector}
              onChange={(e) => setSelectedSector(e.target.value)}
              className="w-full rounded-md border border-[#4a5568] bg-[#2d3748] px-3 py-2 text-sm text-[#e2e8f0] focus:border-blue-500 focus:outline-none"
            >
              {SECTORS.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
          </div>
          <Button
            onClick={handleStartScan}
            loading={createJobMutation.isPending}
            disabled={!!activeJobId && (activeJob?.status === "running" || activeJob?.status === "pending")}
          >
            스캔 시작
          </Button>
        </div>

        {!isAuthenticated && (
          <p className="text-xs text-[#718096]">
            스캔을 실행하려면{" "}
            <button
              onClick={() => setShowAuthModal(true)}
              className="text-blue-400 hover:underline"
            >
              로그인
            </button>
            이 필요합니다.
          </p>
        )}
      </Card>

      {/* 진행 중 상태 */}
      {activeJob && (activeJob.status === "running" || activeJob.status === "pending") && (
        <div id="scan-progress" className="space-y-2 scroll-mt-24">
          <ScannerProgress
            job={activeJob}
            onProgress={handleProgress}
            onComplete={handleComplete}
          />
          {lastMessage && (
            <p className="text-xs text-[#718096] font-mono">{lastMessage}</p>
          )}
        </div>
      )}

      {/* 스캔 결과 */}
      {results && results.length > 0 && (
        <div id="scan-results" className="scroll-mt-24">
          <ScanResults results={results} />
        </div>
      )}

      {/* 최근 작업 목록 */}
      {isAuthenticated && (
        <Card id="recent-jobs" className="space-y-3 scroll-mt-24">
          <h2 className="text-base font-semibold text-[#e2e8f0]">최근 스캔 작업</h2>
          {jobsLoading ? (
            <FullPageSpinner />
          ) : jobs && jobs.length > 0 ? (
            <div className="space-y-2">
              {jobs.map((job) => (
                <div
                  key={job.id}
                  className="flex items-center justify-between rounded-md border border-[#2d3748] p-3 cursor-pointer hover:border-[#4a5568] transition-colors"
                  onClick={() => setActiveJobId(job.id)}
                >
                  <div className="flex items-center gap-3">
                    <Badge variant={statusToBadge(job.status) as "warning" | "success" | "danger" | "default"}>
                      {job.status}
                    </Badge>
                    <span className="text-sm text-[#a0aec0]">
                      {job.sector ?? "전체"}
                    </span>
                  </div>
                  <div className="text-xs text-[#718096]">
                    {job.processed}/{job.total ?? "?"} ·{" "}
                    {new Date(job.created_at).toLocaleString("ko-KR")}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-[#718096]">스캔 기록이 없습니다.</p>
          )}
        </Card>
      )}

      </div>

      {/* 빠른 이동 목차 (TOC) */}
      <div className="hidden lg:block w-48 shrink-0 sticky top-24 self-start">
        <div className="flex flex-col gap-2 p-4 rounded-xl bg-[#1a202c]/60 border border-[#2d3748]/60 backdrop-blur-md shadow-lg">
          <span className="text-xs font-bold text-[#a0aec0] mb-1 uppercase tracking-wider flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line><line x1="3" y1="6" x2="3.01" y2="6"></line><line x1="3" y1="12" x2="3.01" y2="12"></line><line x1="3" y1="18" x2="3.01" y2="18"></line></svg>
            빠른 이동
          </span>
          <a href="#scan-settings" className="text-sm text-[#718096] hover:text-[#3b82f6] hover:bg-[#2d3748]/30 px-2 py-1.5 rounded-md transition-all">스캔 설정</a>
          {activeJob && (activeJob.status === "running" || activeJob.status === "pending") && (
            <a href="#scan-progress" className="text-sm text-[#718096] hover:text-[#3b82f6] hover:bg-[#2d3748]/30 px-2 py-1.5 rounded-md transition-all">진행 상태</a>
          )}
          {results && results.length > 0 && (
            <a href="#scan-results" className="text-sm text-[#718096] hover:text-[#3b82f6] hover:bg-[#2d3748]/30 px-2 py-1.5 rounded-md transition-all">스캔 결과</a>
          )}
          {isAuthenticated && (
            <a href="#recent-jobs" className="text-sm text-[#718096] hover:text-[#3b82f6] hover:bg-[#2d3748]/30 px-2 py-1.5 rounded-md transition-all">최근 작업 목록</a>
          )}
        </div>
      </div>

      {showAuthModal && <AuthModal onClose={() => setShowAuthModal(false)} />}
    </div>
  );
}
