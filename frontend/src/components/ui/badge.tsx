import { type HTMLAttributes } from "react";

type BadgeVariant = "buy" | "sell" | "hold" | "default" | "success" | "warning" | "danger";

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
}

const variantClasses: Record<BadgeVariant, string> = {
  buy: "bg-emerald-950/75 text-[#00c853] border-[#00c853]/45 shadow-[0_0_12px_rgba(0,200,83,0.18)]",
  sell: "bg-red-950/75 text-[#ff1744] border-[#ff1744]/45 shadow-[0_0_12px_rgba(255,23,68,0.18)]",
  hold: "bg-amber-950/75 text-amber-400 border-amber-500/45 shadow-[0_0_12px_rgba(245,158,11,0.15)]",
  success: "bg-emerald-950/75 text-[#00c853] border-[#00c853]/45 shadow-[0_0_12px_rgba(0,200,83,0.18)]",
  warning: "bg-amber-950/75 text-amber-400 border-amber-500/45 shadow-[0_0_12px_rgba(245,158,11,0.15)]",
  danger: "bg-red-950/75 text-[#ff1744] border-[#ff1744]/45 shadow-[0_0_12px_rgba(255,23,68,0.18)]",
  default: "bg-[#2d3748]/60 text-[#a0aec0] border-[#4a5568]/60",
};

export function Badge({ variant = "default", className = "", children, ...props }: BadgeProps) {
  return (
    <span
      className={[
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold",
        variantClasses[variant],
        className,
      ].join(" ")}
      {...props}
    >
      {children}
    </span>
  );
}

export function signalToBadgeVariant(signal: string): BadgeVariant {
  const s = signal.toUpperCase();
  if (s === "BUY" || s === "STRONG BUY") return "buy";
  if (s === "SELL" || s === "STRONG SELL") return "sell";
  if (s === "HOLD") return "hold";
  return "default";
}
