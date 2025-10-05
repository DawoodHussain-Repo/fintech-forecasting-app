"use client";

import Link from "next/link";
import { ArrowLeft, Star } from "lucide-react";
import { DataTable } from "@/components/data-table";
import { Button } from "@/components/ui/button";
import { useWatchlist } from "@/hooks/use-watchlist";

export default function WatchlistPage() {
  const { items, removeItem } = useWatchlist();

  return (
    <div className="relative space-y-12">
      {/* Glow Background Accent */}
      <div className="absolute -top-32 -left-32 w-80 h-80 bg-primary/10 blur-[120px] rounded-full pointer-events-none" />
      <div className="absolute -bottom-32 -right-32 w-80 h-80 bg-primary/5 blur-[120px] rounded-full pointer-events-none" />

      {/* Header Section */}
      <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between relative z-10">
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-[0.25em] text-primary/80">
            Your Watchlist
          </p>
          <h1 className="text-4xl font-extrabold tracking-tight text-foreground neon-glow">
            Favorites & Quick Access
          </h1>
          <p className="text-muted-foreground text-sm max-w-xl">
            Keep track of the assets that matter most. Your starred stocks,
            crypto, and forex pairs live here for instant monitoring.
          </p>
        </div>

        <Link href="/dashboard">
          <Button
            variant="ghost"
            className="gap-2 hover:bg-primary/10 hover:text-primary transition-all"
          >
            <ArrowLeft className="h-4 w-4" /> Back to Dashboard
          </Button>
        </Link>
      </div>

      {/* Watchlist Panel */}
      <div className="glass border border-primary/20 rounded-2xl p-8 shadow-lg relative z-10">
        <div className="mb-6 flex items-center gap-3 text-sm text-muted-foreground">
          <Star className="h-5 w-5 text-amber-400 drop-shadow-glow" />
          <span className="text-foreground/80">
            Add items from the dashboard to keep them starred here.
          </span>
        </div>

        {items.length > 0 ? (
          <DataTable items={items} onRemove={removeItem} />
        ) : (
          <div className="flex flex-col items-center justify-center text-center py-20">
            <Star className="h-12 w-12 text-amber-400/70 mb-4" />
            <p className="text-lg font-medium text-foreground">
              No items in your watchlist yet
            </p>
            <p className="text-sm text-muted-foreground max-w-sm">
              Browse the dashboard and click the star icon on a stock, crypto,
              or forex pair to save it here.
            </p>
            <Link href="/dashboard">
              <Button className="mt-6 px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:glow-md transition-all">
                Go to Dashboard
              </Button>
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}
