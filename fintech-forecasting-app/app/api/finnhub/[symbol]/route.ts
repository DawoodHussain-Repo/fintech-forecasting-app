import { NextResponse } from "next/server";

const FINNHUB_API_KEY =
  process.env.FINNHUB_API_KEY || "d3gss2pr01qpep687uggd3gss2pr01qpep687uh0";
const FINNHUB_BASE = "https://finnhub.io/api/v1";

class ApiError extends Error {
  status: number;
  constructor(message: string, status = 502) {
    super(message);
    this.status = status;
  }
}

async function fetchFinnhubQuote(symbol: string) {
  const url = new URL(`${FINNHUB_BASE}/quote`);
  url.searchParams.set("symbol", symbol);
  url.searchParams.set("token", FINNHUB_API_KEY);

  const response = await fetch(url.toString(), { cache: "no-store" });
  if (!response.ok) {
    throw new ApiError(`Finnhub quote failed`, response.status);
  }

  const data = await response.json();
  if (data.error) throw new ApiError(String(data.error), 400);

  return {
    symbol,
    price: data.c || 0,
    changePercent: data.dp || 0,
    open: data.o || 0,
    high: data.h || 0,
    low: data.l || 0,
    previousClose: data.pc || 0,
    volume: 0,
    latestTradingDay: new Date(data.t * 1000 || Date.now())
      .toISOString()
      .split("T")[0],
    currency: "USD",
  };
}

async function fetchYfinanceCandles(symbol: string, range: string = "1M") {
  const backendUrl = process.env.BACKEND_URL || "http://localhost:5000";
  const url = new URL(`${backendUrl}/api/candles/${symbol}`);
  url.searchParams.set("range", range);

  try {
    const response = await fetch(url.toString(), { cache: "no-store" });
    if (!response.ok) return [];
    const data = await response.json();
    return data.candles || [];
  } catch {
    return [];
  }
}

export async function GET(
  request: Request,
  { params }: { params: { symbol: string } }
) {
  try {
    const resolvedParams = await params;
    const { searchParams } = new URL(request.url);
    const range = searchParams.get("range") ?? "1M";
    const symbol = resolvedParams.symbol.trim().toUpperCase();

    if (!symbol) {
      return NextResponse.json({ error: "Symbol required" }, { status: 400 });
    }

    const [quote, candles] = await Promise.all([
      fetchFinnhubQuote(symbol),
      fetchYfinanceCandles(symbol, range),
    ]);

    return NextResponse.json({ quote, candles });
  } catch (error) {
    if (error instanceof ApiError) {
      return NextResponse.json(
        { error: error.message },
        { status: error.status }
      );
    }
    console.error("API error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
