"use client";

import { useCallback } from "react";
import { useLocalStorage } from "@/hooks/use-local-storage";
import type { WatchlistRow } from "@/components/data-table";

const STORAGE_KEY = "fintech-forecast-watchlist";

export function useWatchlist(initial: WatchlistRow[] = []) {
  const [items, setItems] = useLocalStorage<WatchlistRow[]>(
    STORAGE_KEY,
    initial
  );

  const addItem = useCallback(
    (item: WatchlistRow) => {
      setItems((current) => {
        const existing = current.filter(
          (entry) => entry.symbol !== item.symbol
        );
        return [...existing, item];
      });
    },
    [setItems]
  );

  const removeItem = useCallback(
    (symbol: string) => {
      setItems((current) => current.filter((entry) => entry.symbol !== symbol));
    },
    [setItems]
  );

  return { items, addItem, removeItem };
}
