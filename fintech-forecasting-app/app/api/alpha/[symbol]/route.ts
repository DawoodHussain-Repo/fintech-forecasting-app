import { NextResponse } from "next/server";
import { normalizeQuote } from "@/lib/alpha-vantage";

const API_KEY = process.env.ALPHA_VANTAGE_API_KEY ?? "demo";
const API_BASE = "https://www.alphavantage.co/query";

interface AlphaSeries {
  [date: string]: {
    "1. open": string | number;
    "2. high": string | number;
    "3. low": string | number;
    "4. close": string | number;
    "5. volume": string | number;
  };
}

type AlphaRawSeries = Record<string, Record<string, string>>;

type Instrument =
  | { type: "stock"; symbol: string }
  | { type: "crypto"; symbol: string; market: string }
  | { type: "forex"; from: string; to: string };

class AlphaVantageError extends Error {
  status: number;

  constructor(message: string, status = 502) {
    super(message);
    this.status = status;
  }
}

async function fetchAlpha(
  params: Record<string, string>
): Promise<Record<string, unknown>> {
  const url = new URL(API_BASE);
  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, value);
  }
  url.searchParams.set("apikey", API_KEY);

  const response = await fetch(url.toString(), {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new AlphaVantageError(
      "Alpha Vantage request failed",
      response.status
    );
  }

  const json = (await response.json()) as Record<string, unknown>;

  if (json.Note) {
    throw new AlphaVantageError(
      "Alpha Vantage rate limit reached. Please try again shortly.",
      429
    );
  }

  if (json["Error Message"]) {
    throw new AlphaVantageError(String(json["Error Message"]), 400);
  }

  return json;
}

function detectInstrument(symbol: string): Instrument {
  const trimmed = symbol.trim().toUpperCase();
  if (!trimmed) {
    throw new AlphaVantageError("Symbol is required", 400);
  }

  if (trimmed.includes("/")) {
    const [from, to] = trimmed.split("/");
    if (!from || !to) {
      throw new AlphaVantageError(
        "Invalid forex pair format. Use FROM/TO, e.g., EUR/USD",
        400
      );
    }
    return { type: "forex", from, to };
  }

  if (/^[A-Z]{1,5}$/.test(trimmed)) {
    return { type: "stock", symbol: trimmed };
  }

  return { type: "crypto", symbol: trimmed, market: "USD" };
}

function mapStockSeries(raw: AlphaRawSeries | undefined): AlphaSeries {
  if (!raw) return {};
  return Object.fromEntries(
    Object.entries(raw).map(([date, values]) => [
      date,
      {
        "1. open": values["1. open"],
        "2. high": values["2. high"],
        "3. low": values["3. low"],
        "4. close": values["4. close"],
        "5. volume": values["6. volume"] ?? values["5. volume"] ?? 0,
      },
    ])
  );
}

function mapForexSeries(raw: AlphaRawSeries | undefined): AlphaSeries {
  if (!raw) return {};
  return Object.fromEntries(
    Object.entries(raw).map(([date, values]) => [
      date,
      {
        "1. open": values["1. open"],
        "2. high": values["2. high"],
        "3. low": values["3. low"],
        "4. close": values["4. close"],
        "5. volume": 0,
      },
    ])
  );
}

function mapCryptoSeries(raw: AlphaRawSeries | undefined): AlphaSeries {
  if (!raw) return {};
  return Object.fromEntries(
    Object.entries(raw).map(([date, values]) => [
      date,
      {
        "1. open": values["1a. open (USD)"] ?? values["1. open"],
        "2. high": values["2a. high (USD)"] ?? values["2. high"],
        "3. low": values["3a. low (USD)"] ?? values["3. low"],
        "4. close": values["4a. close (USD)"] ?? values["4. close"],
        "5. volume": values["5. volume"] ?? 0,
      },
    ])
  );
}

function limitSeries(series: AlphaSeries, limit = 120): AlphaSeries {
  const entries = Object.entries(series).sort(
    (a, b) => new Date(b[0]).getTime() - new Date(a[0]).getTime()
  );
  return Object.fromEntries(entries.slice(0, limit));
}

function deriveQuote(
  symbol: string,
  candles: AlphaSeries,
  currency: string
): Record<string, string> {
  const entries = Object.entries(candles).sort(
    (a, b) => new Date(b[0]).getTime() - new Date(a[0]).getTime()
  );

  if (entries.length === 0) return {};

  const [latestDate, latestValues] = entries[0];
  const previousValues = entries[1]?.[1] ?? latestValues;

  const latestClose = Number(latestValues["4. close"]) || 0;
  const previousClose = Number(previousValues["4. close"]) || latestClose;
  const changePercent = previousClose
    ? ((latestClose - previousClose) / previousClose) * 100
    : 0;

  return {
    "01. symbol": symbol,
    "02. open": String(latestValues["1. open"] ?? 0),
    "03. high": String(latestValues["2. high"] ?? 0),
    "04. low": String(latestValues["3. low"] ?? 0),
    "05. price": String(latestClose),
    "06. volume": String(latestValues["5. volume"] ?? 0),
    "07. latest trading day": latestDate,
    "08. previous close": String(previousClose),
    "09. change": String(latestClose - previousClose),
    "10. change percent": `${changePercent.toFixed(2)}%`,
    currency,
  };
}

async function fetchStock(symbol: string) {
  const [seriesJson, quoteJson] = await Promise.all([
    fetchAlpha({
      function: "TIME_SERIES_DAILY_ADJUSTED",
      symbol,
      outputsize: "compact",
    }),
    fetchAlpha({ function: "GLOBAL_QUOTE", symbol }),
  ]);

  const series = limitSeries(
    mapStockSeries(seriesJson["Time Series (Daily)"] as AlphaRawSeries)
  );
  const quote =
    normalizeQuote(
      (quoteJson["Global Quote"] as Record<string, string>) ?? {}
    ) ?? normalizeQuote(deriveQuote(symbol, series, "USD"));

  return { quote, candles: series };
}

async function fetchForex(from: string, to: string) {
  const [seriesJson, rateJson] = await Promise.all([
    fetchAlpha({
      function: "FX_DAILY",
      from_symbol: from,
      to_symbol: to,
      outputsize: "compact",
    }),
    fetchAlpha({
      function: "CURRENCY_EXCHANGE_RATE",
      from_currency: from,
      to_currency: to,
    }),
  ]);

  const series = limitSeries(
    mapForexSeries(seriesJson["Time Series FX (Daily)"] as AlphaRawSeries)
  );
  const quoteSource = rateJson["Realtime Currency Exchange Rate"] as
    | Record<string, string>
    | undefined;

  let derived = normalizeQuote(
    quoteSource
      ? {
          "01. symbol": `${from}/${to}`,
          "05. price": quoteSource["5. Exchange Rate"] ?? "0",
          "07. latest trading day": quoteSource["6. Last Refreshed"],
          currency: to,
        }
      : {}
  );

  if (!derived) {
    derived = normalizeQuote(deriveQuote(`${from}/${to}`, series, to));
  }

  return { quote: derived, candles: series };
}

async function fetchCrypto(symbol: string, market: string) {
  const [seriesJson, quoteJson] = await Promise.all([
    fetchAlpha({ function: "DIGITAL_CURRENCY_DAILY", symbol, market }),
    fetchAlpha({ function: "GLOBAL_QUOTE", symbol }),
  ]);

  const series = limitSeries(
    mapCryptoSeries(
      seriesJson["Time Series (Digital Currency Daily)"] as AlphaRawSeries
    )
  );

  let quote = normalizeQuote(
    (quoteJson["Global Quote"] as Record<string, string>) ?? {}
  );
  if (!quote) {
    quote = normalizeQuote(deriveQuote(symbol, series, market));
  }

  return { quote, candles: series };
}

export async function GET(
  _request: Request,
  { params }: { params: { symbol: string } }
) {
  try {
    const instrument = detectInstrument(params.symbol);

    switch (instrument.type) {
      case "stock": {
        const payload = await fetchStock(instrument.symbol);
        return NextResponse.json(payload);
      }
      case "forex": {
        const payload = await fetchForex(instrument.from, instrument.to);
        return NextResponse.json(payload);
      }
      case "crypto": {
        const payload = await fetchCrypto(instrument.symbol, instrument.market);
        return NextResponse.json(payload);
      }
      default:
        return NextResponse.json(
          { error: "Unsupported instrument" },
          { status: 400 }
        );
    }
  } catch (error) {
    if (error instanceof AlphaVantageError) {
      return NextResponse.json(
        { error: error.message },
        { status: error.status }
      );
    }

    console.error("Alpha Vantage API error", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
