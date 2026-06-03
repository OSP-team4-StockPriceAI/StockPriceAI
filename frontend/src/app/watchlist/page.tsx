"use client";

import { useState } from "react";
import { WatchlistTable } from "@/components/watchlist/WatchlistTable";
import { AuthModal } from "@/components/layout/AuthModal";
import { Button } from "@/components/ui/button";
import { useAuthStore } from "@/store/auth";

export default function WatchlistPage() {
  const { isAuthenticated } = useAuthStore();
  const [showAuthModal, setShowAuthModal] = useState(false);

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-[#e2e8f0]">관심종목</h1>
          <p className="mt-1 text-sm text-[#718096]">
            등록한 종목의 현재가와 AI 신호를 한눈에 확인합니다
          </p>
        </div>
      </div>

      {isAuthenticated ? (
        <WatchlistTable />
      ) : (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <div className="text-5xl mb-4">⭐</div>
          <h3 className="text-lg font-medium text-[#a0aec0]">
            관심종목을 관리하려면 로그인이 필요합니다
          </h3>
          <p className="mt-2 text-sm text-[#718096]">
            로그인 후 종목을 추가하고 실시간 신호를 확인하세요.
          </p>
          <Button className="mt-6" onClick={() => setShowAuthModal(true)}>
            로그인 / 회원가입
          </Button>
        </div>
      )}

      {showAuthModal && <AuthModal onClose={() => setShowAuthModal(false)} />}
    </div>
  );
}
