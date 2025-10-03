"use client";

import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { ArrowLeft, RefreshCcw } from "lucide-react";
import Link from "next/link";
import { ForecastCard } from "@/components/forecast-card";
import { SearchBar } from "@/components/search-bar";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  fetchMarketSnapshot,
  sliceCandlesForRange,
  type RangeOption,
} from "@/lib/alpha-vantage";
import type { MarketSnapshot } from "@/lib/types";
import { HorizonOption, buildDummyForecast } from "@/lib/forecast";
import { formatChangePercent, formatCurrency } from "@/lib/utils";

const rangeForForecast: RangeOption = "6M";

export default function ForecastPage() {
  const searchParams = useSearchParams();
  const initialSymbol = searchParams.get("symbol")?.toUpperCase() ?? "AAPL";

  const [symbol, setSymbol] = useState(initialSymbol);
  const [snapshot, setSnapshot] = useState<MarketSnapshot>({
    quote: null,
    candles: [],
  });
  const [horizon, setHorizon] = useState<HorizonOption>("24h");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        setLoading(true);
        setError(null);
        const data = await fetchMarketSnapshot(symbol);
        if (!active) return;
        setSnapshot(data);
      } catch (err) {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Unable to fetch data");
      } finally {
        if (!active) return;
        setLoading(false);
      }
    }

    load();

    return () => {
      active = false;
    };
  }, [symbol, reloadKey]);

  const candlesForForecast = useMemo(
    () => sliceCandlesForRange(snapshot.candles, rangeForForecast),
    [snapshot]
  );

  const forecastSeries = useMemo(
    () => buildDummyForecast(candlesForForecast, horizon),
    [candlesForForecast, horizon]
  );

  const quote = snapshot.quote;

  return (
    <div className="space-y-12">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <p className="text-sm uppercase tracking-[0.3em] text-muted-foreground">
            Forecast
          </p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight text-foreground">
            Scenario planning with dummy ML output
          </h1>
        </div>
        <Link
          href="/dashboard"
          className="inline-flex items-center gap-2 rounded-full border border-border/60 px-5 py-3 text-sm font-medium text-foreground hover:bg-muted/40"
        >
          <ArrowLeft className="h-4 w-4" /> Back to Dashboard
        </Link>
      </div>

      <Card className="border-none bg-card/80">
        <CardHeader className="space-y-2">
          <CardTitle>Configure forecast</CardTitle>
          <CardDescription>
            Symbols pull live data from Alpha Vantage. Forecasts are placeholder
            trends until ML models are added.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <SearchBar
            placeholder="Select symbol for forecasting"
            onSearch={(value) => setSymbol(value)}
            loading={loading}
          />

          {quote ? (
            <div className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
              <Badge
                intent={
                  quote.changePercent >= 0
                    ? "success"
                    : quote.changePercent < 0
                    ? "danger"
                    : "default"
                }
              >
                {formatChangePercent(quote.changePercent)}
              </Badge>
              <span>
                {quote.symbol} ·{" "}
                {formatCurrency(quote.price, quote.currency ?? "USD")} · Volume{" "}
                {quote.volume.toLocaleString()}
              </span>
            </div>
          ) : null}

          {error ? (
            <div className="rounded-xl border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-500">
              {error}
            </div>
          ) : null}

          <Button
            variant="ghost"
            className="gap-2"
            onClick={() => setReloadKey((key) => key + 1)}
            disabled={loading}
          >
            <RefreshCcw className="h-4 w-4" /> Refresh data
          </Button>
        </CardContent>
      </Card>

      <ForecastCard
        symbol={symbol}
        candles={candlesForForecast}
        forecast={forecastSeries}
        horizon={horizon}
        onHorizonChange={setHorizon}
        currency={quote?.currency ?? "USD"}
      />
    </div>
  );
}
