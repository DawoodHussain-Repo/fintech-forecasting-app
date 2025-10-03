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
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn, formatCurrency } from "@/lib/utils";
import type { Candle } from "@/lib/types";
import type { ChartData, ChartOptions } from "chart.js";
import type { HorizonOption } from "@/lib/forecast";

ChartJS.register(
  TimeScale,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  CandlestickController,
  CandlestickElement
);

interface ForecastPoint {
  timestamp: string;
  value: number;
}

interface ForecastCardProps {
  symbol: string;
  candles: Candle[];
  forecast: ForecastPoint[];
  horizon: HorizonOption;
  onHorizonChange: (horizon: HorizonOption) => void;
  currency?: string;
}

const horizons: HorizonOption[] = ["1h", "3h", "24h", "72h"];

export function ForecastCard({
  symbol,
  candles,
  forecast,
  horizon,
  onHorizonChange,
  currency = "USD",
}: ForecastCardProps) {
  const combined = useMemo<ChartData<"candlestick" | "line">>(() => {
    return {
      datasets: [
        {
          type: "candlestick" as const,
          label: `${symbol} history`,
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
          borderWidth: 1,
        },
        {
          type: "line" as const,
          label: `${symbol} forecast (${horizon})`,
          data: forecast.map((point) => ({
            x: new Date(point.timestamp),
            y: point.value,
          })),
          borderColor: "#fbbf24",
          borderWidth: 2,
          borderDash: [6, 6],
          pointRadius: 2,
          tension: 0.3,
        },
      ],
    } as unknown as ChartData<"candlestick" | "line">;
  }, [candles, forecast, horizon, symbol]);

  const options = useMemo<ChartOptions<"candlestick" | "line">>(() => {
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
            label(context: TooltipItem<"candlestick" | "line">) {
              if (context.dataset.type === "line") {
                const value = context.parsed.y;
                return `Forecast • ${formatCurrency(value, currency)}`;
              }
              const candle = context.raw as {
                o: number;
                h: number;
                l: number;
                c: number;
              };
              return `Close ${formatCurrency(candle.c, currency)}`;
            },
          },
        },
      },
      scales: {
        x: {
          type: "time" as const,
          grid: { display: false },
          ticks: { color: "hsl(var(--muted-foreground))" },
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
    <Card className="border-none bg-gradient-to-br from-background via-background/90 to-background shadow-glow">
      <CardHeader className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <CardTitle className="text-2xl">Forecast preview</CardTitle>
          <CardDescription>
            Forecast coming soon – powered by ML models (ARIMA, LSTM,
            Transformers)
          </CardDescription>
        </div>
        <div className="flex items-center gap-1 rounded-full bg-card/70 p-1 shadow-inner">
          {horizons.map((option) => (
            <Button
              key={option}
              variant="ghost"
              className={cn(
                "rounded-full px-4 text-xs font-semibold uppercase tracking-wide",
                horizon === option
                  ? "bg-secondary text-secondary-foreground shadow"
                  : "text-muted-foreground hover:bg-muted/40"
              )}
              onClick={() => onHorizonChange(option)}
            >
              {option}
            </Button>
          ))}
        </div>
      </CardHeader>
      <CardContent className="h-[360px]">
        {candles.length === 0 ? (
          <div className="flex h-full items-center justify-center text-muted-foreground">
            Select a symbol to preview forecast trends.
          </div>
        ) : (
          <Chart
            type="candlestick"
            data={combined}
            options={options}
            updateMode="resize"
          />
        )}
      </CardContent>
    </Card>
  );
}
