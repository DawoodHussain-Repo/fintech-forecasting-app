"use client";

import Link from "next/link";
import { ArrowLeft, Star } from "lucide-react";
import { DataTable } from "@/components/data-table";
import { Button } from "@/components/ui/button";
import { useWatchlist } from "@/hooks/use-watchlist";

export default function WatchlistPage() {
  const { items, removeItem } = useWatchlist();

  return (
    <div className="space-y-10">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <p className="text-sm uppercase tracking-[0.3em] text-muted-foreground">
            Watchlist
          </p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight text-foreground">
            Saved stocks, crypto, and forex pairs
          </h1>
        </div>
        <Link href="/dashboard">
          <Button variant="ghost" className="gap-2">
            <ArrowLeft className="h-4 w-4" /> Back to Dashboard
          </Button>
        </Link>
      </div>

      <div className="rounded-3xl border border-border/60 bg-card/80 p-6 shadow-sm backdrop-blur">
        <div className="mb-6 flex items-center gap-3 text-sm text-muted-foreground">
          <Star className="h-4 w-4 text-amber-500" />
          Items you add from the dashboard will appear here for quick
          monitoring.
        </div>
        <DataTable items={items} onRemove={removeItem} />
      </div>
    </div>
  );
}
