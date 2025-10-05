"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "@/components/theme-toggle";
import { cn } from "@/lib/utils";
import { TrendingUp } from "lucide-react";

const links = [
  { href: "/", label: "Home" },
  { href: "/dashboard", label: "Dashboard" },
  { href: "/forecast", label: "Forecast" },
  { href: "/watchlist", label: "Watchlist" },
  { href: "/about", label: "About" },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 glass">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-8 px-8 py-4">
        <Link href="/" className="flex items-center gap-3 group">
          <div className="relative flex h-10 w-10 items-center justify-center rounded-lg bg-black/40 border border-primary/20 group-hover:border-primary/40 transition-all duration-300">
            <TrendingUp className="w-5 h-5 text-primary" />
          </div>

          <div className="flex flex-col leading-tight">
            <span className="text-lg font-bold text-primary">
              FinTech Forecaster
            </span>
            <span className="text-xs font-normal text-muted-foreground">
              AI-Powered Market Intelligence
            </span>
          </div>
        </Link>

        <nav className="hidden md:flex items-center gap-1 bg-black/40 px-2 py-2 rounded-md border border-primary/10 mt-1">
          {links.map((link) => {
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  "relative px-4 py-1.5 text-sm font-medium transition-all duration-200 rounded",
                  isActive
                    ? "text-primary"
                    : "text-muted-foreground hover:text-primary/80"
                )}
              >
                <span className="relative z-10">{link.label}</span>
                {isActive && (
                  <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-3/4 h-0.5 bg-primary rounded-full" />
                )}
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center gap-4">
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
