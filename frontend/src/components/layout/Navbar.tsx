"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuthStore } from "@/store/auth";
import { Button } from "@/components/ui/button";
import { AuthModal } from "@/components/layout/AuthModal";

const navLinks = [
  { href: "/", label: "대시보드" },
  { href: "/scanner", label: "스캐너" },
  { href: "/watchlist", label: "관심종목" },
];

export function Navbar() {
  const pathname = usePathname();
  const { isAuthenticated, user, logout } = useAuthStore();
  const [showAuthModal, setShowAuthModal] = useState(false);

  return (
    <>
      <header className="sticky top-0 z-50 border-b border-[#2d3748] bg-[#0f1117]/90 backdrop-blur">
        <nav className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-6">
            <Link href="/" className="flex items-center gap-2">
              <span className="text-lg font-bold text-blue-400">StockPriceAI</span>
            </Link>
            <div className="hidden items-center gap-1 sm:flex">
              {navLinks.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className={[
                    "rounded-md px-3 py-2 text-sm font-medium transition-colors",
                    pathname === link.href
                      ? "bg-[#2d3748] text-[#e2e8f0]"
                      : "text-[#718096] hover:text-[#e2e8f0]",
                  ].join(" ")}
                >
                  {link.label}
                </Link>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-3">
            {isAuthenticated ? (
              <>
                <span className="hidden text-sm text-[#718096] sm:block">
                  {user?.email}
                </span>
                <Button variant="ghost" size="sm" onClick={logout}>
                  로그아웃
                </Button>
              </>
            ) : (
              <Button variant="secondary" size="sm" onClick={() => setShowAuthModal(true)}>
                로그인
              </Button>
            )}
          </div>
        </nav>
      </header>
      {showAuthModal && <AuthModal onClose={() => setShowAuthModal(false)} />}
    </>
  );
}
