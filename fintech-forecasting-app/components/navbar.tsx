"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "@/components/theme-toggle";
import { cn } from "@/lib/utils";

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
    <header className="sticky top-0 z-50 glass shadow-lg">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-8 px-8 py-4">
        <Link
          href="/"
          className="flex items-center gap-3 text-2xl font-extrabold tracking-tight bg-gradient-to-r from-blue-400 via-purple-400 to-green-400 bg-clip-text text-transparent drop-shadow-lg"
        >
          <span className="relative flex h-12 w-12 items-center justify-center overflow-hidden rounded-full gradient-main shadow-lg border-2 border-white/30">
            <span className="absolute inset-0 animate-shimmer bg-[linear-gradient(120deg,rgba(255,255,255,0)_0%,rgba(255,255,255,.7)_50%,rgba(255,255,255,0)_100%)] bg-[length:200%_100%] rounded-full" />
            <span className="relative text-white text-xl font-bold">FF</span>
          </span>
          <div className="flex flex-col leading-tight">
            <span className="text-lg font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              FinTech Forecaster
            </span>
            <span className="text-xs font-normal text-muted-foreground">
              Alpha insights at a glance
            </span>
          </div>
        </Link>

        <nav className="hidden md:flex items-center gap-2 rounded-full border border-border/60 bg-card/80 px-3 py-2 text-base shadow-md glass">
          {links.map((link) => {
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  "rounded-full px-5 py-2 font-semibold transition-all duration-200 hover:bg-gradient-to-r hover:from-blue-500/20 hover:to-purple-500/20 hover:text-foreground focus-visible:ring-2 focus-visible:ring-primary/60",
                  isActive
                    ? "bg-gradient-to-r from-blue-500/40 to-purple-500/40 text-primary-foreground shadow-md"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {link.label}
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
