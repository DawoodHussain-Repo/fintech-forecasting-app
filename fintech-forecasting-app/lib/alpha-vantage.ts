import type { Candle, MarketSnapshot } from "@/lib/types";

export type RangeOption = "1D" | "1W" | "1M" | "6M" | "1Y";

const RANGE_TO_DAYS: Record<RangeOption, number> = {
  "1D": 1,
  "1W": 7,
  "1M": 30,
  "6M": 182,
  "1Y": 365,
};

export function sliceCandlesForRange(
  candles: Candle[],
  range: RangeOption
): Candle[] {
  const days = RANGE_TO_DAYS[range];
  if (candles.length === 0) return candles;

  const cutoff = new Date(candles[candles.length - 1].timestamp);
  cutoff.setDate(cutoff.getDate() - days + 1);

  return candles.filter((candle) => new Date(candle.timestamp) >= cutoff);
}

export async function fetchMarketSnapshot(
  symbol: string
): Promise<MarketSnapshot> {
  const response = await fetch(`/api/finnhub/${encodeURIComponent(symbol)}`);

  if (!response.ok) {
    // Try to get error message from response
    try {
      const errorData = await response.json();
      throw new Error(errorData.error || "Failed to fetch market data");
    } catch {
      throw new Error("Failed to fetch market data");
    }
  }

  const json = await response.json();

  return {
    quote: json.quote,
    candles: json.candles || [],
  };
}

export function describeRange(range: RangeOption) {
  switch (range) {
    case "1D":
      return "1 Day";
    case "1W":
      return "1 Week";
    case "1M":
      return "1 Month";
    case "6M":
      return "6 Months";
    case "1Y":
      return "1 Year";
    default:
      return range;
  }
}
