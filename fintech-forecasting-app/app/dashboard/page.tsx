"use client";

import { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { gsap } from "gsap";
import { LiquidGlassStockCard } from "@/components/liquid-glass-stock-card";
import { NeonSearchBar } from "@/components/neon-search-bar";
import { AlertCircle, X, TrendingUp } from "lucide-react";
import { fetchMarketSnapshot } from "@/lib/alpha-vantage";
import type { MarketSnapshot } from "@/lib/types";

// Popular stocks to display
const FEATURED_STOCKS = [
  "AAPL", // Apple
  "GOOGL", // Google
  "MSFT", // Microsoft
  "TSLA", // Tesla
  "AMZN", // Amazon
  "NVDA", // NVIDIA
];

const SEARCH_SUGGESTIONS = [
  "AAPL",
  "GOOGL",
  "MSFT",
  "TSLA",
  "AMZN",
  "NVDA",
  "META",
  "NFLX",
  "BTC",
  "ETH",
  "EUR/USD",
  "GBP/USD",
];

interface StockData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap?: string;
}

export default function DashboardPage() {
  const router = useRouter();
  const [stocks, setStocks] = useState<StockData[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchLoading, setSearchLoading] = useState(false);
  const [showToast, setShowToast] = useState(false);
  const [toastMessage, setToastMessage] = useState("");
  const headerRef = useRef<HTMLDivElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Animate header on mount
    if (headerRef.current) {
      gsap.fromTo(
        headerRef.current,
        { opacity: 0, y: -30 },
        { opacity: 1, y: 0, duration: 0.8, ease: "power3.out" }
      );
    }

    loadFeaturedStocks();
  }, []);

  const loadFeaturedStocks = async () => {
    setLoading(true);
    const stockData: StockData[] = [];

    for (const symbol of FEATURED_STOCKS) {
      try {
        const data = await fetchMarketSnapshot(symbol);
        if (data.quote) {
          const previousClose = data.quote.previousClose || data.quote.price;
          const change = data.quote.price - previousClose;

          stockData.push({
            symbol: data.quote.symbol || symbol,
            price: data.quote.price,
            change: change,
            changePercent: data.quote.changePercent,
            volume: data.quote.volume,
          });
        }
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : "Error";

        // Show toast for rate limit only once
        if (errorMsg.includes("rate limit") && !showToast) {
          setToastMessage("⚠️ API Rate Limit Reached. Showing cached data.");
          setShowToast(true);
          setTimeout(() => setShowToast(false), 10000);
        }

        // Add placeholder data
        stockData.push({
          symbol: symbol,
          price: 0,
          change: 0,
          changePercent: 0,
          volume: 0,
        });
      }
    }

    setStocks(stockData);
    setLoading(false);
  };

  const handleSearch = async (searchSymbol: string) => {
    setSearchLoading(true);
    try {
      const data = await fetchMarketSnapshot(searchSymbol);
      if (data.quote) {
        // Navigate to detailed view
        router.push(`/forecast?symbol=${encodeURIComponent(searchSymbol)}`);
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Error searching";
      setToastMessage(errorMsg);
      setShowToast(true);
      setTimeout(() => setShowToast(false), 5000);
    } finally {
      setSearchLoading(false);
    }
  };

  const handleStockClick = (symbol: string) => {
    router.push(`/forecast?symbol=${encodeURIComponent(symbol)}`);
  };

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute top-0 left-0 w-96 h-96 bg-primary/10 rounded-full blur-[120px] animate-pulse" />
      <div
        className="absolute bottom-0 right-0 w-96 h-96 bg-primary/5 rounded-full blur-[120px] animate-pulse"
        style={{ animationDelay: "1s" }}
      />

      {/* Toast Notification */}
      {showToast && (
        <div className="fixed top-4 right-4 z-50 max-w-md animate-in slide-in-from-top-5">
          <div className="glass neon-pulse">
            <div className="flex items-start gap-3 p-4">
              <AlertCircle className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
              <div className="flex-1">
                <p className="text-sm font-medium text-primary">
                  API Rate Limit
                </p>
                <p className="text-sm text-foreground/80 mt-1">
                  {toastMessage}
                </p>
              </div>
              <button
                onClick={() => setShowToast(false)}
                className="text-foreground/70 hover:text-primary transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="relative z-10 container mx-auto px-4 py-12 space-y-12">
        {/* Header */}
        <div ref={headerRef} className="text-center space-y-6">
          <div className="inline-block">
            <h1 className="text-6xl font-bold text-primary mb-2 tracking-tight">
              FINTECH FORECASTER
            </h1>
            <div className="h-0.5 bg-gradient-to-r from-transparent via-primary to-transparent" />
          </div>

          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Real-time market intelligence powered by{" "}
            <span className="text-primary font-semibold">AI predictions</span>
          </p>

          {/* Search Bar */}
          <div className="pt-6">
            <NeonSearchBar
              onSearch={handleSearch}
              suggestions={SEARCH_SUGGESTIONS}
              loading={searchLoading}
              placeholder="Search stocks (AAPL), crypto (BTC), forex (EUR/USD)..."
            />
          </div>
        </div>

        {/* Featured Stocks Section */}
        <div className="space-y-6">
          <div className="flex items-center gap-3">
            <TrendingUp className="w-6 h-6 text-primary" />
            <h2 className="text-3xl font-bold text-primary">Featured Stocks</h2>
          </div>

          {loading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[...Array(6)].map((_, i) => (
                <div key={i} className="glass h-48 animate-pulse">
                  <div className="p-6 space-y-4">
                    <div className="h-6 bg-primary/20 rounded w-1/3" />
                    <div className="h-8 bg-primary/20 rounded w-2/3" />
                    <div className="h-4 bg-primary/20 rounded w-1/2" />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div
              ref={gridRef}
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
            >
              {stocks.map((stock, index) => (
                <LiquidGlassStockCard
                  key={stock.symbol}
                  symbol={stock.symbol}
                  price={stock.price}
                  change={stock.change}
                  changePercent={stock.changePercent}
                  volume={stock.volume}
                  marketCap={stock.marketCap}
                  onClick={() => handleStockClick(stock.symbol)}
                  index={index}
                />
              ))}
            </div>
          )}
        </div>

        {/* CTA Section */}
        <div className="glass text-center p-12 relative overflow-hidden group">
          <div className="absolute inset-0 bg-gradient-to-r from-primary/0 via-primary/10 to-primary/0 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
          <div className="relative z-10 space-y-4">
            <h3 className="text-2xl font-bold text-foreground">
              Want to see detailed forecasts?
            </h3>
            <p className="text-muted-foreground max-w-xl mx-auto">
              Click on any stock to view advanced analytics, AI-powered
              predictions, and historical data
            </p>
            <button
              onClick={() => router.push("/forecast")}
              className="mt-4 px-8 py-3 bg-primary text-primary-foreground rounded-lg font-semibold hover:glow-md transition-all duration-300 transform hover:scale-105"
            >
              Explore Forecast Workspace
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
