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
    <header className="sticky top-0 z-40 border-b border-border/60 bg-background/75 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-6 px-6 py-4">
        <Link
          href="/"
          className="flex items-center gap-2 text-lg font-semibold"
        >
          <span className="relative flex h-9 w-9 items-center justify-center overflow-hidden rounded-full bg-primary text-primary-foreground shadow-md">
            <span className="absolute inset-0 animate-shimmer bg-[linear-gradient(120deg,rgba(255,255,255,0)_0%,rgba(255,255,255,.7)_50%,rgba(255,255,255,0)_100%)] bg-[length:200%_100%]" />
            <span className="relative">FF</span>
          </span>
          <div className="flex flex-col leading-tight">
            <span>FinTech Forecaster</span>
            <span className="text-xs font-normal text-muted-foreground">
              Alpha insights at a glance
            </span>
          </div>
        </Link>

        <nav className="hidden items-center gap-1 rounded-full border border-border/60 bg-card/70 px-2 py-1 text-sm shadow-sm md:flex">
          {links.map((link) => {
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  "rounded-full px-4 py-2 font-medium transition-colors",
                  isActive
                    ? "bg-primary text-primary-foreground shadow"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center gap-3">
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
