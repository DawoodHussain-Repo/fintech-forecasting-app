"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { LineChart, Plus, AlertCircle, X, ChevronRight } from "lucide-react";
import { SearchBar } from "@/components/search-bar";
import { ChartCard } from "@/components/chart-card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { MetricCard } from "@/components/metric-card";
import {
  fetchMarketSnapshot,
  RangeOption,
  sliceCandlesForRange,
} from "@/lib/alpha-vantage";
import {
  formatChangePercent,
  formatCompactNumber,
  formatCurrency,
} from "@/lib/utils";
import type { MarketSnapshot } from "@/lib/types";
import { useWatchlist } from "@/hooks/use-watchlist";

const DEFAULT_SYMBOL = "AAPL";
const DEFAULT_RANGE: RangeOption = "1M";

export default function DashboardPage() {
  const [symbol, setSymbol] = useState(DEFAULT_SYMBOL);
  const [range, setRange] = useState<RangeOption>(DEFAULT_RANGE);
  const [snapshot, setSnapshot] = useState<MarketSnapshot>({
    quote: null,
    candles: [],
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showToast, setShowToast] = useState(false);
  const [toastMessage, setToastMessage] = useState("");

  const { addItem } = useWatchlist();

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
        const errorMsg =
          err instanceof Error ? err.message : "Unable to fetch data";
        setError(errorMsg);

        // Show toast for rate limit errors
        if (
          errorMsg.includes("rate limit") ||
          errorMsg.includes("25 requests")
        ) {
          setToastMessage(
            "⚠️ API Rate Limit Reached (25 requests/day). Please upgrade your API key or try again tomorrow."
          );
          setShowToast(true);
          // Auto-hide toast after 10 seconds
          setTimeout(() => setShowToast(false), 10000);
        }
      } finally {
        if (!active) return;
        setLoading(false);
      }
    }

    load();

    return () => {
      active = false;
    };
  }, [symbol]);

  const displayCandles = useMemo(
    () => sliceCandlesForRange(snapshot.candles, range),
    [snapshot, range]
  );
  const quote = snapshot.quote;

  const changeIntent = quote
    ? quote.changePercent > 0
      ? "success"
      : quote.changePercent < 0
      ? "danger"
      : "default"
    : "default";

  const addToWatchlist = () => {
    if (!quote || displayCandles.length === 0) return;
    addItem({
      symbol: quote.symbol || symbol,
      price: quote.price,
      changePercent: quote.changePercent,
      sparkline: displayCandles.slice(-20).map((c) => c.close),
      currency: quote.currency,
    });
  };

  return (
    <div className="space-y-12">
      {/* Toast Notification */}
      {showToast && (
        <div className="fixed top-4 right-4 z-50 max-w-md animate-in slide-in-from-top-5">
          <div className="rounded-lg border border-orange-500/50 bg-orange-500/10 p-4 shadow-lg backdrop-blur-sm">
            <div className="flex items-start gap-3">
              <AlertCircle className="h-5 w-5 text-orange-500 mt-0.5 flex-shrink-0" />
              <div className="flex-1">
                <p className="text-sm font-medium text-orange-500">
                  API Rate Limit Reached
                </p>
                <p className="text-sm text-orange-500/90 mt-1">
                  {toastMessage}
                </p>
              </div>
              <button
                onClick={() => setShowToast(false)}
                className="text-orange-500/70 hover:text-orange-500 transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      )}

      <section className="space-y-6">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-sm uppercase tracking-[0.3em] text-muted-foreground">
              Dashboard
            </p>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight text-foreground">
              Alpha Vantage market intelligence
            </h1>
          </div>
          <Link
            href="/forecast"
            className="inline-flex items-center gap-2 rounded-full border border-border/60 px-6 py-3 text-sm font-medium text-foreground hover:bg-muted/40"
          >
            Forecast workspace
            <ChevronRight className="h-4 w-4" />
          </Link>
        </div>

        <SearchBar
          placeholder="Enter stock, crypto, or forex symbol (e.g., AAPL, BTC, EUR/USD)"
          onSearch={(value) => setSymbol(value)}
          loading={loading}
        />

        {error ? (
          <div className="rounded-2xl border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-500">
            {error}
          </div>
        ) : null}

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <MetricCard
            label="Current price"
            value={
              quote ? formatCurrency(quote.price, quote.currency ?? "USD") : "—"
            }
            hint={
              quote?.latestTradingDay
                ? `As of ${quote.latestTradingDay}`
                : "Awaiting data"
            }
            icon={<LineChart className="h-4 w-4 text-muted-foreground" />}
          />
          <MetricCard
            label="Change"
            value={quote ? formatChangePercent(quote.changePercent) : "—"}
            hint="vs previous close"
            icon={
              <Badge intent={changeIntent}>
                {quote ? formatChangePercent(quote.changePercent) : "—"}
              </Badge>
            }
          />
          <MetricCard
            label="Volume"
            value={quote ? formatCompactNumber(quote.volume) : "—"}
            hint="Latest trading session"
          />
          <MetricCard
            label="Day range"
            value={
              quote
                ? `${formatCurrency(
                    quote.low,
                    quote.currency ?? "USD"
                  )} – ${formatCurrency(quote.high, quote.currency ?? "USD")}`
                : "—"
            }
            hint={
              quote
                ? `Open ${formatCurrency(quote.open, quote.currency ?? "USD")}`
                : "Awaiting data"
            }
          />
        </div>
      </section>

      <section className="space-y-6">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight text-foreground">
              Price action
            </h2>
            <p className="text-sm text-muted-foreground">
              Candlestick chart powered by Alpha Vantage daily series.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              className="gap-2"
              onClick={addToWatchlist}
              disabled={!quote || loading}
            >
              <Plus className="h-4 w-4" />
              Add to Watchlist
            </Button>
            <Link href={`/forecast?symbol=${encodeURIComponent(symbol)}`}>
              <Button className="gap-2" variant="secondary">
                Forecast
                <ChevronRight className="h-4 w-4" />
              </Button>
            </Link>
          </div>
        </div>

        <ChartCard
          symbol={symbol}
          candles={displayCandles}
          currency={quote?.currency ?? "USD"}
          range={range}
          onRangeChange={setRange}
          loading={loading}
        />
      </section>
    </div>
  );
}
