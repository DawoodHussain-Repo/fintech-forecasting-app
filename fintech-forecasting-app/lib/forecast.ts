import type { Candle } from "@/lib/types";

export type HorizonOption = "1h" | "3h" | "24h" | "72h";

const HORIZON_TO_POINTS: Record<HorizonOption, number> = {
  "1h": 4,
  "3h": 6,
  "24h": 8,
  "72h": 10,
};

export function buildDummyForecast(candles: Candle[], horizon: HorizonOption) {
  if (candles.length === 0) return [];

  const steps = HORIZON_TO_POINTS[horizon] ?? 6;
  const lastCandle = candles[candles.length - 1];
  const baseDate = new Date(lastCandle.timestamp);
  const basePrice = lastCandle.close;

  return Array.from({ length: steps }, (_, index) => {
    const dayOffset = index + 1;
    const nextDate = new Date(baseDate);
    nextDate.setDate(baseDate.getDate() + dayOffset);
    const seasonal = Math.sin((dayOffset / steps) * Math.PI * 1.5);
    const drift = 0.004 * dayOffset;
    const value = basePrice * (1 + seasonal * 0.01 + drift);

    return {
      timestamp: nextDate.toISOString(),
      value: Number(value.toFixed(2)),
    };
  });
}
