import type { Candle, MarketSnapshot, Quote } from "@/lib/types";

type AlphaCandle = Record<string, string | number>;
type AlphaSeries = Record<string, AlphaCandle>;

export type RangeOption = "1D" | "1W" | "1M" | "6M" | "1Y";

const RANGE_TO_DAYS: Record<RangeOption, number> = {
  "1D": 1,
  "1W": 7,
  "1M": 30,
  "6M": 182,
  "1Y": 365,
};

export function normalizeCandles(
  raw: AlphaSeries | undefined | null
): Candle[] {
  if (!raw) return [];

  return Object.entries(raw)
    .map(([date, candle]) => ({
      timestamp: `${date}T00:00:00Z`,
      open: Number(candle["1. open"] ?? candle.open ?? 0),
      high: Number(candle["2. high"] ?? candle.high ?? 0),
      low: Number(candle["3. low"] ?? candle.low ?? 0),
      close: Number(candle["4. close"] ?? candle.close ?? 0),
      volume: Number(candle["5. volume"] ?? candle.volume ?? 0),
    }))
    .filter((candle) => !Number.isNaN(candle.close))
    .sort(
      (a, b) =>
        new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );
}

export function normalizeQuote(
  raw: Record<string, string | undefined>
): Quote | null {
  if (!raw) return null;

  const price = Number(raw["05. price"] ?? raw.price);
  if (!Number.isFinite(price)) return null;

  return {
    symbol: raw["01. symbol"] ?? raw.symbol ?? "",
    price,
    changePercent: Number(
      (raw["10. change percent"] ?? raw.change_percent ?? "0")
        .toString()
        .replace("%", "")
    ),
    open: Number(raw["02. open"] ?? raw.open ?? 0),
    high: Number(raw["03. high"] ?? raw.high ?? 0),
    low: Number(raw["04. low"] ?? raw.low ?? 0),
    previousClose: Number(raw["08. previous close"] ?? raw.previous_close ?? 0),
    volume: Number(raw["06. volume"] ?? raw.volume ?? 0),
    latestTradingDay:
      raw["07. latest trading day"] ?? raw.latest_trading_day ?? "",
    currency: raw.currency ?? "USD",
  };
}

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
  const response = await fetch(`/api/alpha/${encodeURIComponent(symbol)}`);

  if (!response.ok) {
    throw new Error("Failed to fetch market data");
  }

  const json = await response.json();
  const candles = normalizeCandles(json.candles);

  return {
    quote: normalizeQuote(json.quote),
    candles,
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
