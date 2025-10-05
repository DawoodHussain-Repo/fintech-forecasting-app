"use client";

import { useMemo } from "react";
import type { TooltipItem } from "chart.js";
import {
  Chart as ChartJS,
  CategoryScale,
  Legend,
  LinearScale,
  TimeScale,
  Tooltip,
} from "chart.js";
import {
  CandlestickController,
  CandlestickElement,
} from "chartjs-chart-financial";
import { Chart } from "react-chartjs-2";
import "chartjs-adapter-date-fns";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Candle } from "@/lib/types";
import { describeRange, type RangeOption } from "@/lib/alpha-vantage";
import { cn, formatCurrency } from "@/lib/utils";

ChartJS.register(
  TimeScale,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  CandlestickController,
  CandlestickElement
);

const ranges: RangeOption[] = ["1D", "1W", "1M", "6M", "1Y"];

interface ChartCardProps {
  symbol: string;
  candles: Candle[];
  currency?: string;
  range: RangeOption;
  onRangeChange: (range: RangeOption) => void;
  loading?: boolean;
}

export function ChartCard({
  symbol,
  candles,
  currency = "USD",
  range,
  onRangeChange,
  loading,
}: ChartCardProps) {
  const data = useMemo(() => {
    return {
      datasets: [
        {
          label: `${symbol} • ${describeRange(range)}`,
          data: candles.map((candle) => ({
            x: new Date(candle.timestamp),
            o: candle.open,
            h: candle.high,
            l: candle.low,
            c: candle.close,
          })),
          color: {
            up: "#34d399",
            down: "#f87171",
            unchanged: "#60a5fa",
          },
          borderColor: "#3b82f6",
          borderWidth: 1,
        },
      ],
    };
  }, [candles, symbol, range]);

  const options = useMemo(() => {
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        intersect: false,
        mode: "index" as const,
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label(context: TooltipItem<"candlestick">) {
              const candle = context.raw as {
                o: number;
                h: number;
                l: number;
                c: number;
              };
              const close = formatCurrency(candle.c, currency);
              return `${close} · O ${formatCurrency(
                candle.o,
                currency
              )} · H ${formatCurrency(candle.h, currency)} · L ${formatCurrency(
                candle.l,
                currency
              )}`;
            },
          },
        },
      },
      scales: {
        x: {
          type: "time" as const,
          time: {
            tooltipFormat: "MMM dd, yyyy",
          },
          ticks: {
            color: "hsl(var(--muted-foreground))",
          },
          grid: {
            display: false,
          },
        },
        y: {
          position: "right" as const,
          ticks: {
            color: "hsl(var(--muted-foreground))",
            callback(value: number | string) {
              const numeric = typeof value === "string" ? Number(value) : value;
              return formatCurrency(numeric, currency);
            },
          },
          grid: {
            color: "hsla(var(--border), 0.2)",
          },
        },
      },
    };
  }, [currency]);

  return (
    <Card className="border-none bg-gradient-to-br from-background/60 via-background/80 to-background shadow-glow">
      <CardHeader className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <CardTitle className="flex flex-col gap-1 text-2xl">
          <span>{symbol}</span>
          <span className="text-sm font-normal text-muted-foreground">
            Interactive candlestick chart powered by Finnhub
          </span>
        </CardTitle>
        <div className="flex items-center gap-1 rounded-full bg-card/70 p-1 shadow-inner">
          {ranges.map((option) => (
            <Button
              key={option}
              variant="ghost"
              className={cn(
                "rounded-full px-4 text-xs font-semibold uppercase tracking-wide",
                range === option
                  ? "bg-primary text-primary-foreground shadow"
                  : "text-muted-foreground hover:bg-muted/40"
              )}
              onClick={() => onRangeChange(option)}
              disabled={loading}
            >
              {option}
            </Button>
          ))}
        </div>
      </CardHeader>
      <CardContent className="h-[420px]">
        {candles.length === 0 ? (
          <div className="flex h-full items-center justify-center text-muted-foreground">
            {loading ? "Loading chart..." : "No data available for this range."}
          </div>
        ) : (
          <Chart
            type="candlestick"
            data={data}
            options={options}
            updateMode="resize"
          />
        )}
      </CardContent>
    </Card>
  );
}
