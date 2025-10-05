"use client";

import { useEffect, useRef, useState } from "react";
import { gsap } from "gsap";
import { TrendingUp, TrendingDown, Star } from "lucide-react";
import { formatCurrency, formatChangePercent } from "@/lib/utils";
import { useWatchlist } from "@/hooks/use-watchlist";

interface LiquidGlassStockCardProps {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume?: number;
  marketCap?: string;
  onClick?: () => void;
  index?: number;
}

export function LiquidGlassStockCard({
  symbol,
  price,
  change,
  changePercent,
  volume,
  marketCap,
  onClick,
  index = 0,
}: LiquidGlassStockCardProps) {
  const cardRef = useRef<HTMLDivElement>(null);
  const isPositive = changePercent >= 0;
  const { items, addItem, removeItem } = useWatchlist();
  const [isInWatchlist, setIsInWatchlist] = useState(false);

  useEffect(() => {
    // Check if already in watchlist
    setIsInWatchlist(items.some((item) => item.symbol === symbol));
  }, [items, symbol]);

  useEffect(() => {
    if (cardRef.current) {
      // Animate card entrance
      gsap.fromTo(
        cardRef.current,
        {
          opacity: 0,
          y: 30,
        },
        {
          opacity: 1,
          y: 0,
          duration: 0.5,
          delay: index * 0.08,
          ease: "power2.out",
        }
      );
    }
  }, [index]);

  const handleWatchlistToggle = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();

    if (isInWatchlist) {
      removeItem(symbol);
    } else {
      addItem({
        symbol,
        price,
        changePercent,
        sparkline: [],
        currency: "USD",
      });
    }
  };

  const handleCardClick = (e: React.MouseEvent) => {
    // Only trigger onClick if not clicking on the star button
    if (onClick && e.currentTarget === e.target) {
      onClick();
    } else if (onClick && !(e.target as HTMLElement).closest("button")) {
      onClick();
    }
  };

  return (
    <div
      ref={cardRef}
      className="glass group cursor-pointer relative overflow-hidden"
      onClick={handleCardClick}
    >
      {/* Card content */}
      <div className="p-6 relative z-10">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1">
            <h3 className="text-2xl font-bold text-primary tracking-wide">
              {symbol}
            </h3>
            <p className="text-xs text-muted-foreground mt-1">
              {marketCap || "Market Cap: N/A"}
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleWatchlistToggle}
              className={`p-2 rounded-lg transition-all ${
                isInWatchlist
                  ? "bg-primary/20 text-primary"
                  : "bg-black/40 text-muted-foreground hover:text-primary hover:bg-primary/10"
              }`}
              title={
                isInWatchlist ? "Remove from watchlist" : "Add to watchlist"
              }
            >
              <Star
                className={`w-4 h-4 ${isInWatchlist ? "fill-current" : ""}`}
              />
            </button>

            <div
              className={`p-2 rounded-lg ${
                isPositive ? "bg-primary/10" : "bg-destructive/10"
              }`}
            >
              {isPositive ? (
                <TrendingUp className="w-5 h-5 text-primary" />
              ) : (
                <TrendingDown className="w-5 h-5 text-destructive" />
              )}
            </div>
          </div>
        </div>

        {/* Price */}
        <div className="mb-4">
          <div className="text-3xl font-bold text-foreground mb-1">
            {formatCurrency(price, "USD")}
          </div>

          {/* Change */}
          <div className="flex items-center gap-2">
            <span
              className={`text-sm font-medium ${
                isPositive ? "text-primary" : "text-destructive"
              }`}
            >
              {isPositive ? "+" : ""}
              {formatCurrency(change, "USD")}
            </span>
            <span
              className={`text-sm font-medium px-2 py-0.5 rounded ${
                isPositive
                  ? "bg-primary/20 text-primary"
                  : "bg-destructive/20 text-destructive"
              }`}
            >
              {formatChangePercent(changePercent)}
            </span>
          </div>
        </div>

        {/* Volume */}
        {volume !== undefined && volume > 0 && (
          <div className="pt-4 border-t border-primary/10">
            <p className="text-xs text-muted-foreground">Volume</p>
            <p className="text-sm font-medium text-foreground">
              {(volume / 1000000).toFixed(2)}M
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
