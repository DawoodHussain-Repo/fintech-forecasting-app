"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  ArrowLeft,
  TrendingUp,
  TrendingDown,
  Loader as LoaderIcon,
  BarChart3,
} from "lucide-react";
import { fetchMarketSnapshot, type RangeOption } from "@/lib/alpha-vantage";
import type { MarketSnapshot } from "@/lib/types";
import CandlestickChart from "@/components/candlestick-chart";
import { Button } from "@/components/ui/button";

export default function StockPage() {
  const params = useParams();
  const router = useRouter();
  const symbol = (params.symbol as string).toUpperCase();

  const [data, setData] = useState<MarketSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedRange, setSelectedRange] = useState<RangeOption>("1M");
  const [forecastData, setForecastData] = useState<{
    model_type?: string;
    forecast?: { predicted_price?: number; trend?: string };
    metrics?: { confidence?: number };
  } | null>(null);
  const [forecastLoading, setForecastLoading] = useState(false);

  const loadStockData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const snapshot = await fetchMarketSnapshot(symbol);
      setData(snapshot);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load data");
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  useEffect(() => {
    loadStockData();
  }, [symbol, selectedRange, loadStockData]);

  const handleGenerateForecast = async (modelType: string) => {
    setForecastLoading(true);
    try {
      const response = await fetch("/api/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol,
          model_type: modelType,
          horizon_hours: 72,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to generate forecast");
      }

      const result = await response.json();
      setForecastData(result);
    } catch (err) {
      console.error("Forecast error:", err);
    } finally {
      setForecastLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <LoaderIcon className="h-12 w-12 animate-spin text-primary mx-auto mb-4" />
          <p className="text-lg text-muted-foreground">Loading {symbol}...</p>
        </div>
      </div>
    );
  }

  if (error || !data || !data.quote) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center glass p-8 rounded-lg max-w-md">
          <p className="text-lg text-destructive mb-4">
            {error || "No data available"}
          </p>
          <Button onClick={() => router.push("/dashboard")} variant="outline">
            <ArrowLeft className="mr-2 h-4 w-4" /> Back to Dashboard
          </Button>
        </div>
      </div>
    );
  }

  const { quote, candles } = data;
  const isPositive = quote.changePercent >= 0;

  return (
    <div className="min-h-screen bg-background py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6 flex items-center gap-4">
          <Button
            onClick={() => router.push("/dashboard")}
            variant="ghost"
            size="sm"
            className="text-primary hover:text-primary/80"
          >
            <ArrowLeft className="mr-2 h-4 w-4" /> Dashboard
          </Button>
        </div>

        {/* Price Card */}
        <div className="glass rounded-lg p-6 mb-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h1 className="text-4xl font-bold text-primary mb-2">{symbol}</h1>
              <p className="text-sm text-muted-foreground">
                Last updated: {quote.latestTradingDay}
              </p>
            </div>
            <div className="text-right">
              <div className="text-5xl font-bold mb-2">
                ${quote.price.toFixed(2)}
              </div>
              <div
                className={`flex items-center justify-end gap-2 text-lg font-semibold ${
                  isPositive ? "text-green-500" : "text-red-500"
                }`}
              >
                {isPositive ? (
                  <TrendingUp className="h-5 w-5" />
                ) : (
                  <TrendingDown className="h-5 w-5" />
                )}
                <span>
                  {isPositive ? "+" : ""}
                  {quote.changePercent.toFixed(2)}%
                </span>
              </div>
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6 pt-6 border-t border-primary/10">
            <div>
              <p className="text-xs text-muted-foreground mb-1">Open</p>
              <p className="text-lg font-semibold text-primary">
                ${quote.open.toFixed(2)}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground mb-1">High</p>
              <p className="text-lg font-semibold text-primary">
                ${quote.high.toFixed(2)}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground mb-1">Low</p>
              <p className="text-lg font-semibold text-primary">
                ${quote.low.toFixed(2)}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground mb-1">Prev Close</p>
              <p className="text-lg font-semibold text-primary">
                ${quote.previousClose.toFixed(2)}
              </p>
            </div>
          </div>
        </div>

        {/* Chart Section */}
        <div className="glass rounded-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold text-primary flex items-center gap-2">
              <BarChart3 className="h-6 w-6" />
              Price Chart
            </h2>
            <div className="flex gap-2">
              {(["1D", "1W", "1M", "6M", "1Y"] as RangeOption[]).map(
                (range) => (
                  <Button
                    key={range}
                    onClick={() => setSelectedRange(range)}
                    variant={selectedRange === range ? "default" : "outline"}
                    size="sm"
                    className={
                      selectedRange === range ? "bg-primary text-black" : ""
                    }
                  >
                    {range}
                  </Button>
                )
              )}
            </div>
          </div>

          {candles && candles.length > 0 ? (
            <CandlestickChart
              historicalData={candles}
              forecastData={[]}
              symbol={symbol}
              loading={loading}
            />
          ) : (
            <div className="h-96 flex items-center justify-center text-muted-foreground">
              <p>No chart data available</p>
            </div>
          )}
        </div>

        {/* Forecast Section */}
        <div className="glass rounded-lg p-6">
          <h2 className="text-2xl font-bold text-primary mb-4">
            AI Forecasting
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <Button
              onClick={() => handleGenerateForecast("moving_average")}
              disabled={forecastLoading}
              className="bg-primary/10 hover:bg-primary/20 text-primary border border-primary/20"
            >
              {forecastLoading ? (
                <LoaderIcon className="mr-2 h-4 w-4 animate-spin" />
              ) : null}
              Moving Average
            </Button>
            <Button
              onClick={() => handleGenerateForecast("arima")}
              disabled={forecastLoading}
              className="bg-primary/10 hover:bg-primary/20 text-primary border border-primary/20"
            >
              {forecastLoading ? (
                <LoaderIcon className="mr-2 h-4 w-4 animate-spin" />
              ) : null}
              ARIMA Model
            </Button>
            <Button
              onClick={() => handleGenerateForecast("lstm")}
              disabled={forecastLoading}
              className="bg-primary/10 hover:bg-primary/20 text-primary border border-primary/20"
            >
              {forecastLoading ? (
                <LoaderIcon className="mr-2 h-4 w-4 animate-spin" />
              ) : null}
              LSTM Neural Network
            </Button>
          </div>

          {forecastData && (
            <div className="border border-primary/20 rounded-lg p-4 bg-primary/5">
              <h3 className="font-semibold text-lg mb-2 text-primary">
                Forecast Results
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-xs text-muted-foreground">Model</p>
                  <p className="font-semibold text-primary">
                    {forecastData.model_type}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">
                    Predicted Price
                  </p>
                  <p className="font-semibold text-primary">
                    $
                    {forecastData.forecast?.predicted_price?.toFixed(2) ||
                      "N/A"}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Confidence</p>
                  <p className="font-semibold text-primary">
                    {forecastData.metrics?.confidence?.toFixed(1) || "N/A"}%
                  </p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Trend</p>
                  <p className="font-semibold text-primary">
                    {forecastData.forecast?.trend || "N/A"}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
