"use client";

import { useState, type FormEvent } from "react";
import { Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface SearchBarProps {
  placeholder?: string;
  onSearch: (symbol: string) => void;
  loading?: boolean;
}

export function SearchBar({
  placeholder = "Search symbol...",
  onSearch,
  loading,
}: SearchBarProps) {
  const [value, setValue] = useState("");

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = value.trim().toUpperCase();
    if (trimmed.length === 0) return;
    onSearch(trimmed);
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex w-full flex-col gap-3 rounded-2xl border border-border/60 bg-card/70 p-3 shadow-sm backdrop-blur md:flex-row md:items-center"
    >
      <div className="flex flex-1 items-center gap-3 rounded-xl bg-background/60 px-3">
        <Search className="h-4 w-4 text-muted-foreground" />
        <Input
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder={placeholder}
          className="border-none bg-transparent px-0 text-base focus-visible:ring-0"
        />
      </div>
      <Button type="submit" className="w-full md:w-auto" disabled={loading}>
        {loading ? "Fetching..." : "Fetch"}
      </Button>
    </form>
  );
}
