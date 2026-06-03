"use client";

interface SkeletonLoaderProps {
  type?: "card-grid" | "chart" | "table" | "technical" | "tab-content";
  count?: number;
}

export function SkeletonLoader({ type = "card-grid", count = 4 }: SkeletonLoaderProps) {
  if (type === "card-grid") {
    return (
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 w-full">
        {Array.from({ length: count }).map((_, idx) => (
          <div key={idx} className="glass-card p-4 flex flex-col justify-between h-[90px]">
            <div className="h-3 w-1/2 rounded animate-skeleton" />
            <div className="h-6 w-3/4 rounded mt-3 animate-skeleton" />
          </div>
        ))}
      </div>
    );
  }

  if (type === "chart") {
    return (
      <div className="glass-card p-4 w-full flex flex-col space-y-4 min-h-[400px]">
        {/* Chart Header */}
        <div className="flex justify-between items-center">
          <div className="h-4 w-1/4 rounded animate-skeleton" />
          <div className="h-6 w-1/3 rounded animate-skeleton" />
        </div>
        {/* Chart Body Grid */}
        <div className="flex-1 w-full rounded animate-skeleton min-h-[300px]" />
      </div>
    );
  }

  if (type === "technical") {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
        {Array.from({ length: 2 }).map((_, idx) => (
          <div key={idx} className="glass-card p-6 flex flex-col items-center justify-center space-y-4 min-h-[220px]">
            {/* Header label */}
            <div className="h-3 w-1/3 rounded animate-skeleton" />
            {/* Gauge Shape Mock */}
            <div className="relative w-40 h-20 overflow-hidden mt-4">
              <div className="absolute top-0 left-0 w-40 h-40 rounded-full border-[12px] border-dashed border-[#2d3748] opacity-40 animate-skeleton" />
            </div>
            {/* Value Label */}
            <div className="h-6 w-16 rounded mt-2 animate-skeleton" />
          </div>
        ))}
      </div>
    );
  }

  if (type === "table") {
    return (
      <div className="w-full space-y-3">
        <div className="flex space-x-4 border-b border-[#2d3748]/50 pb-3">
          <div className="h-4 w-1/5 rounded animate-skeleton" />
          <div className="h-4 w-1/4 rounded animate-skeleton" />
          <div className="h-4 w-1/6 rounded animate-skeleton" />
          <div className="h-4 w-1/6 rounded animate-skeleton" />
        </div>
        {Array.from({ length: count }).map((_, idx) => (
          <div key={idx} className="flex space-x-4 py-3 border-b border-[#2d3748]/20 items-center">
            <div className="h-4 w-1/5 rounded animate-skeleton" />
            <div className="h-3 w-1/4 rounded animate-skeleton" />
            <div className="h-4 w-1/6 rounded animate-skeleton ml-auto" />
            <div className="h-4 w-1/6 rounded animate-skeleton" />
          </div>
        ))}
      </div>
    );
  }

  if (type === "tab-content") {
    return (
      <div className="space-y-6 w-full">
        {/* Metrics Grid */}
        <SkeletonLoader type="card-grid" count={4} />
        {/* Large Visual Section */}
        <SkeletonLoader type="chart" />
      </div>
    );
  }

  return null;
}
