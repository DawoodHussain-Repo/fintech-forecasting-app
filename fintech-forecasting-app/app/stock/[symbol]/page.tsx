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
import {
  fetchMarketSnapshot,
  type RangeOption,
  sliceCandlesForRange,
} from "@/lib/alpha-vantage";
import type { MarketSnapshot } from "@/lib/types";
import CandlestickChart from "@/components/candlestick-chart";
import { Button } from "@/components/ui/button";

interface BackendForecastPoint {
  timestamp: string;
  predicted_price: number;
  price_range_low: number;
  price_range_high: number;
  confidence: string;
}

interface ShortTermForecast {
  direction: string;
  confidence: string;
  predicted_price: number;
  price_range: [number, number];
}

interface TechnicalIndicators {
  sma_7: number;
  rsi: number;
  bb_upper: number;
  bb_lower: number;
  volatility_pct: number;
  momentum_5d_pct: number;
  support_level: number;
  resistance_level: number;
}

interface ForecastData {
  symbol: string;
  model_type: string;
  current_price: number;
  forecast_1h: ShortTermForecast;
  forecast_4h: ShortTermForecast;
  forecast_24h: ShortTermForecast;
  technical_indicators: TechnicalIndicators;
  forecast: BackendForecastPoint[];
  data_points_used: number;
  data_period: string;
  created_at: string;
  cached: boolean;
  // Legacy fields
  metrics?: { confidence?: number; mse?: number; mae?: number };
}

export default function StockPage() {
  const params = useParams();
  const router = useRouter();
  const symbol = (params.symbol as string).toUpperCase();

  const [data, setData] = useState<MarketSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedRange, setSelectedRange] = useState<RangeOption>("1M");
  const [forecastData, setForecastData] = useState<ForecastData | null>(null);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [forecastHorizon, setForecastHorizon] = useState(24);

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

  // Get filtered candles for the selected range
  const getFilteredCandles = useCallback(() => {
    if (!data?.candles) return [];
    return sliceCandlesForRange(data.candles, selectedRange);
  }, [data?.candles, selectedRange]);

  useEffect(() => {
    loadStockData();
  }, [symbol, loadStockData]);

  // Clear forecast data when range changes to avoid confusion
  useEffect(() => {
    setForecastData(null);
  }, [selectedRange]);

  const handleGenerateForecast = async (modelType: string) => {
    setForecastLoading(true);
    setForecastData(null);
    setError(null);
    try {
      const response = await fetch("/api/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol,
          model_type: modelType,
          horizon: forecastHorizon,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to generate forecast");
      }

      const result = await response.json();
      console.log("Forecast result:", result);
      setForecastData(result);
    } catch (err) {
      console.error("Forecast error:", err);
      setError(
        err instanceof Error ? err.message : "Failed to generate forecast"
      );
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

  const { quote } = data;
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

          {data?.candles && data.candles.length > 0 ? (
            <CandlestickChart
              historicalData={getFilteredCandles()}
              forecastData={(forecastData?.forecast || []).map((p) => ({
                timestamp: p.timestamp,
                value: p.predicted_price,
              }))}
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

          {/* Horizon Selection */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-muted-foreground mb-2">
              Forecast Horizon (1-72 Hours)
            </label>
            <div className="flex gap-2 flex-wrap">
              {[6, 12, 24, 48, 72].map((hours) => (
                <Button
                  key={hours}
                  onClick={() => setForecastHorizon(hours)}
                  variant={forecastHorizon === hours ? "default" : "outline"}
                  size="sm"
                  className={
                    forecastHorizon === hours
                      ? "bg-primary text-black"
                      : "border-primary/20 text-primary hover:bg-primary/10"
                  }
                >
                  {hours}h
                </Button>
              ))}
            </div>
          </div>

          {/* Model Selection */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="space-y-2">
              <Button
                onClick={() => handleGenerateForecast("moving_average")}
                disabled={forecastLoading}
                className="w-full bg-primary/10 hover:bg-primary/20 text-primary border border-primary/20"
              >
                {forecastLoading ? (
                  <LoaderIcon className="mr-2 h-4 w-4 animate-spin" />
                ) : null}
                Moving Average
              </Button>
              <p className="text-xs text-muted-foreground text-center">
                Simple trend-following model
              </p>
            </div>

            <div className="space-y-2">
              <Button
                onClick={() => handleGenerateForecast("arima")}
                disabled={forecastLoading}
                className="w-full bg-primary/10 hover:bg-primary/20 text-primary border border-primary/20"
              >
                {forecastLoading ? (
                  <LoaderIcon className="mr-2 h-4 w-4 animate-spin" />
                ) : null}
                ARIMA Model
              </Button>
              <p className="text-xs text-muted-foreground text-center">
                Statistical time series model
              </p>
            </div>

            <div className="space-y-2">
              <Button
                onClick={() => handleGenerateForecast("lstm")}
                disabled={forecastLoading}
                className="w-full bg-primary/10 hover:bg-primary/20 text-primary border border-primary/20"
              >
                {forecastLoading ? (
                  <LoaderIcon className="mr-2 h-4 w-4 animate-spin" />
                ) : null}
                LSTM Neural Network
              </Button>
              <p className="text-xs text-muted-foreground text-center">
                Deep learning model
              </p>
            </div>
          </div>

          {/* Loading State */}
          {forecastLoading && (
            <div className="border border-primary/20 rounded-lg p-6 bg-primary/5 text-center">
              <LoaderIcon className="h-8 w-8 animate-spin text-primary mx-auto mb-2" />
              <p className="text-primary">
                Generating {forecastHorizon}h forecast...
              </p>
            </div>
          )}

          {/* Forecast Results */}
          {forecastData && !forecastLoading && (
            <div className="border border-primary/20 rounded-lg p-6 bg-primary/5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-lg text-primary">
                  Forecast Results
                </h3>
                {forecastData.cached && (
                  <span className="text-xs px-2 py-1 bg-yellow-500/20 text-yellow-300 rounded">
                    Cached
                  </span>
                )}
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                <div>
                  <p className="text-xs text-muted-foreground">Model</p>
                  <p className="font-semibold text-primary capitalize">
                    {forecastData.model_type?.replace("_", " ")}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Horizon</p>
                  <p className="font-semibold text-primary">
                    {forecastHorizon} hours
                  </p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Current Price</p>
                  <p className="font-semibold text-primary">
                    ${forecastData.current_price?.toFixed(2) || "N/A"}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Data Points</p>
                  <p className="font-semibold text-primary">
                    {forecastData.data_points_used || "N/A"}
                  </p>
                </div>
              </div>

              {/* Short-term forecasts */}
              {forecastData.forecast_1h && (
                <div className="mb-4">
                  <h4 className="text-sm font-semibold text-primary mb-2">
                    Short-term Predictions
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    {/* 1h forecast */}
                    <div className="bg-primary/10 rounded p-3">
                      <p className="text-xs text-muted-foreground mb-1">
                        1 Hour
                      </p>
                      <p className="text-lg font-bold text-primary">
                        $
                        {forecastData.forecast_1h?.predicted_price?.toFixed(
                          2
                        ) || "N/A"}
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Direction:{" "}
                        <span className="font-semibold capitalize">
                          {forecastData.forecast_1h?.direction || "N/A"}
                        </span>
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Confidence:{" "}
                        <span className="font-semibold capitalize">
                          {forecastData.forecast_1h?.confidence || "N/A"}
                        </span>
                      </p>
                    </div>

                    {/* 4h forecast */}
                    <div className="bg-primary/10 rounded p-3">
                      <p className="text-xs text-muted-foreground mb-1">
                        4 Hours
                      </p>
                      <p className="text-lg font-bold text-primary">
                        $
                        {forecastData.forecast_4h?.predicted_price?.toFixed(
                          2
                        ) || "N/A"}
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Direction:{" "}
                        <span className="font-semibold capitalize">
                          {forecastData.forecast_4h?.direction || "N/A"}
                        </span>
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Confidence:{" "}
                        <span className="font-semibold capitalize">
                          {forecastData.forecast_4h?.confidence || "N/A"}
                        </span>
                      </p>
                    </div>

                    {/* 24h forecast */}
                    <div className="bg-primary/10 rounded p-3">
                      <p className="text-xs text-muted-foreground mb-1">
                        24 Hours
                      </p>
                      <p className="text-lg font-bold text-primary">
                        $
                        {forecastData.forecast_24h?.predicted_price?.toFixed(
                          2
                        ) || "N/A"}
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Direction:{" "}
                        <span className="font-semibold capitalize">
                          {forecastData.forecast_24h?.direction || "N/A"}
                        </span>
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Confidence:{" "}
                        <span className="font-semibold capitalize">
                          {forecastData.forecast_24h?.confidence || "N/A"}
                        </span>
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Technical Indicators */}
              {forecastData.technical_indicators && (
                <div className="mb-4">
                  <h4 className="text-sm font-semibold text-primary mb-2">
                    Technical Indicators
                  </h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                    <div className="bg-primary/5 rounded p-2">
                      <p className="text-muted-foreground">SMA-7</p>
                      <p className="font-semibold text-primary">
                        $
                        {forecastData.technical_indicators?.sma_7?.toFixed(2) ||
                          "N/A"}
                      </p>
                    </div>
                    <div className="bg-primary/5 rounded p-2">
                      <p className="text-muted-foreground">RSI</p>
                      <p className="font-semibold text-primary">
                        {forecastData.technical_indicators?.rsi?.toFixed(2) ||
                          "N/A"}
                      </p>
                    </div>
                    <div className="bg-primary/5 rounded p-2">
                      <p className="text-muted-foreground">Volatility</p>
                      <p className="font-semibold text-primary">
                        {forecastData.technical_indicators?.volatility_pct?.toFixed(
                          2
                        ) || "N/A"}
                        %
                      </p>
                    </div>
                    <div className="bg-primary/5 rounded p-2">
                      <p className="text-muted-foreground">Momentum (5d)</p>
                      <p className="font-semibold text-primary">
                        {forecastData.technical_indicators?.momentum_5d_pct?.toFixed(
                          2
                        ) || "N/A"}
                        %
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Forecast Preview */}
              {forecastData.forecast && forecastData.forecast.length > 0 && (
                <div className="mt-4">
                  <p className="text-sm text-muted-foreground mb-2">
                    Next few predictions:
                  </p>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    {forecastData.forecast
                      .slice(0, 4)
                      .map((point: BackendForecastPoint, index: number) => (
                        <div key={index} className="bg-primary/10 rounded p-2">
                          <p className="text-xs text-muted-foreground">
                            +{index + 1}h
                          </p>
                          <p className="font-semibold text-primary">
                            ${point.predicted_price?.toFixed(2) || "N/A"}
                          </p>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
