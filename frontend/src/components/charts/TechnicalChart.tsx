"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";
import type { TechnicalResponse } from "@/types/api";
import { FullPageSpinner } from "@/components/ui/spinner";

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => <FullPageSpinner />,
});

interface TechnicalChartProps {
  data: TechnicalResponse;
}

export function TechnicalIndicatorsChart({ data }: TechnicalChartProps) {
  const ind = data.latest_indicators;

  const gaugeTraces = useMemo(() => {
    const rsi = ind.rsi14 ?? 50;
    const stoch_k = ind.stoch_k ?? 50;
    const williams_r = ind.williams_r ?? -50;
    const bb_pos = ind.bb_position ?? 0.5;
    const macd_hist = ind.macd_hist ?? 0;
    const vol_ratio = ind.volume_ratio ?? 1.0;

    return [
      // Row 1
      {
        type: "indicator" as const,
        mode: "gauge+number" as const,
        value: rsi,
        title: { text: "RSI (14)", font: { color: "#a0aec0", size: 15 } },
        gauge: {
          axis: { range: [0, 100], tickcolor: "#718096", tickfont: { color: "#718096" } },
          bar: { color: rsi > 70 ? "#ff1744" : rsi < 30 ? "#00c853" : "#3b82f6" },
          bgcolor: "#1f2937",
          bordercolor: "#2d3748",
          steps: [
            { range: [0, 30], color: "rgba(0, 200, 83, 0.25)" },
            { range: [70, 100], color: "rgba(255, 23, 68, 0.25)" },
          ],
          threshold: { line: { color: "#ffffff", width: 2 }, thickness: 0.75, value: rsi },
        },
        domain: { x: [0, 0.3], y: [0.65, 1] },
        number: { font: { color: "#e2e8f0", size: 24 } },
      },
      {
        type: "indicator" as const,
        mode: "gauge+number" as const,
        value: stoch_k,
        title: { text: "Stoch %K", font: { color: "#a0aec0", size: 15 } },
        gauge: {
          axis: { range: [0, 100], tickcolor: "#718096", tickfont: { color: "#718096" } },
          bar: { color: stoch_k > 80 ? "#ff1744" : stoch_k < 20 ? "#00c853" : "#3b82f6" },
          bgcolor: "#1f2937",
          bordercolor: "#2d3748",
          steps: [
            { range: [0, 20], color: "rgba(0, 200, 83, 0.25)" },
            { range: [80, 100], color: "rgba(255, 23, 68, 0.25)" },
          ],
        },
        domain: { x: [0.35, 0.65], y: [0.65, 1] },
        number: { font: { color: "#e2e8f0", size: 24 } },
      },
      {
        type: "indicator" as const,
        mode: "gauge+number" as const,
        value: williams_r,
        title: { text: "Williams %R", font: { color: "#a0aec0", size: 15 } },
        gauge: {
          axis: { range: [-100, 0], tickcolor: "#718096", tickfont: { color: "#718096" } },
          bar: { color: williams_r > -20 ? "#ff1744" : williams_r < -80 ? "#00c853" : "#3b82f6" },
          bgcolor: "#1f2937",
          bordercolor: "#2d3748",
          steps: [
            { range: [-100, -80], color: "rgba(0, 200, 83, 0.25)" },
            { range: [-20, 0], color: "rgba(255, 23, 68, 0.25)" },
          ],
        },
        domain: { x: [0.7, 1], y: [0.65, 1] },
        number: { font: { color: "#e2e8f0", size: 24 } },
      },
      // Row 2
      {
        type: "indicator" as const,
        mode: "number" as const,
        value: macd_hist,
        title: { text: "MACD Hist", font: { color: "#a0aec0", size: 15 } },
        number: {
          font: { color: macd_hist > 0 ? "#00c853" : "#ff1744", size: 24 },
          valueformat: ".3f"
        },
        domain: { x: [0, 0.3], y: [0, 0.4] },
      },
      {
        type: "indicator" as const,
        mode: "gauge+number" as const,
        value: bb_pos,
        title: { text: "BB Position", font: { color: "#a0aec0", size: 15 } },
        gauge: {
          axis: { range: [-0.2, 1.2], tickcolor: "#718096", tickfont: { color: "#718096" } },
          bar: { color: bb_pos > 0.9 ? "#ff1744" : bb_pos < 0.1 ? "#00c853" : "#9ca3af" },
          bgcolor: "#1f2937",
          bordercolor: "#2d3748",
          steps: [
            { range: [-0.2, 0.1], color: "rgba(0, 200, 83, 0.25)" },
            { range: [0.9, 1.2], color: "rgba(255, 23, 68, 0.25)" },
          ],
        },
        domain: { x: [0.35, 0.65], y: [0, 0.4] },
        number: { font: { color: "#e2e8f0", size: 24 }, valueformat: ".2f" },
      },
      {
        type: "indicator" as const,
        mode: "gauge+number" as const,
        value: vol_ratio,
        title: { text: "Volume Ratio", font: { color: "#a0aec0", size: 15 } },
        gauge: {
          axis: { range: [0, 3], tickcolor: "#718096", tickfont: { color: "#718096" } },
          bar: { color: vol_ratio > 1.5 ? "#f59e0b" : "#3b82f6" },
          bgcolor: "#1f2937",
          bordercolor: "#2d3748",
          steps: [
            { range: [1.5, 3], color: "rgba(245, 158, 11, 0.25)" },
          ],
        },
        domain: { x: [0.7, 1], y: [0, 0.4] },
        number: { font: { color: "#e2e8f0", size: 24 }, valueformat: ".2f" },
      },
    ];
  }, [ind]);

  const layout = useMemo(
    () => ({
      paper_bgcolor: "#1a1f2e",
      plot_bgcolor: "#1a1f2e",
      font: { color: "#a0aec0" },
      margin: { t: 60, r: 20, b: 40, l: 20 },
      height: 480,
      annotations: [
        { x: 0.15, y: 0.58, xref: 'paper' as const, yref: 'paper' as const, text: "30 이하: 과매도 / 70 이상: 과매수", showarrow: false, font: { size: 10, color: "#718096" } },
        { x: 0.5, y: 0.58, xref: 'paper' as const, yref: 'paper' as const, text: "20 이하: 과매도 / 80 이상: 과매수", showarrow: false, font: { size: 10, color: "#718096" } },
        { x: 0.85, y: 0.58, xref: 'paper' as const, yref: 'paper' as const, text: "-80 이하: 과매도 / -20 이상: 과매수", showarrow: false, font: { size: 10, color: "#718096" } },
        { x: 0.15, y: -0.05, xref: 'paper' as const, yref: 'paper' as const, text: "양수: 상승 추세 / 음수: 하락 추세", showarrow: false, font: { size: 10, color: "#718096" } },
        { x: 0.5, y: -0.05, xref: 'paper' as const, yref: 'paper' as const, text: "0 근접: 하단 / 1 근접: 상단", showarrow: false, font: { size: 10, color: "#718096" } },
        { x: 0.85, y: -0.05, xref: 'paper' as const, yref: 'paper' as const, text: "1.5 이상: 거래량 급증", showarrow: false, font: { size: 10, color: "#718096" } }
      ]
    }),
    [],
  );

  return (
    <Plot
      data={gaugeTraces}
      layout={layout}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: "100%" }}
    />
  );
}
