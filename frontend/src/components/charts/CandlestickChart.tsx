"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";
import { FullPageSpinner } from "@/components/ui/spinner";

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => <FullPageSpinner />,
});

interface OHLCVData {
  date: string[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume?: number[];
}

interface CandlestickChartProps {
  data: OHLCVData;
  ticker: string;
}

export function CandlestickChart({ data, ticker }: CandlestickChartProps) {
  const traces = useMemo(() => {
    const candlestick = {
      type: "candlestick" as const,
      x: data.date,
      open: data.open,
      high: data.high,
      low: data.low,
      close: data.close,
      name: ticker,
      increasing: { line: { color: "#00c853" }, fillcolor: "#00c853" },
      decreasing: { line: { color: "#ff1744" }, fillcolor: "#ff1744" },
    };

    if (!data.volume) return [candlestick];

    const volume = {
      type: "bar" as const,
      x: data.date,
      y: data.volume,
      name: "Volume",
      marker: {
        color: data.close.map((c, i) =>
          i === 0 || c >= data.close[i - 1] ? "#00c853" : "#ff1744",
        ),
        opacity: 0.7,
      },
      yaxis: "y2",
    };

    return [candlestick, volume];
  }, [data, ticker]);

  const initialRange = useMemo(() => {
    if (!data.date || data.date.length === 0) return undefined;
    return [
      data.date[0],
      data.date[data.date.length - 1]
    ];
  }, [data.date]);

  const layout = useMemo(
    () => ({
      title: { text: `${ticker} 가격 차트`, font: { color: "#e2e8f0" } },
      paper_bgcolor: "#1a1f2e",
      plot_bgcolor: "#1a1f2e",
      font: { color: "#a0aec0" },
      xaxis: {
        type: "category" as const,
        range: initialRange,
        rangeslider: { visible: true, bordercolor: "#2d3748", bgcolor: "#1f2937" },
        gridcolor: "#2d3748",
        tickfont: { color: "#718096" },
      },
      yaxis: {
        title: { text: "가격 (USD)", font: { color: "#a0aec0" } },
        gridcolor: "#2d3748",
        tickfont: { color: "#718096" },
        domain: [0.35, 1],
      },
      yaxis2: {
        title: { text: "거래량", font: { color: "#a0aec0" } },
        gridcolor: "#2d3748",
        tickfont: { color: "#718096" },
        domain: [0, 0.25],
      },
      margin: { t: 40, r: 20, b: 20, l: 60 },
      showlegend: false,
    }),
    [ticker, initialRange],
  );

  return (
    <Plot
      data={traces}
      layout={layout}
      config={{ responsive: true, displayModeBar: true }}
      style={{ width: "100%", minHeight: "450px" }}
    />
  );
}
