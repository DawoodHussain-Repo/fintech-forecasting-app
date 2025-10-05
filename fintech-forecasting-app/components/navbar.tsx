"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { TrendingUp, Menu, X } from "lucide-react";
import { useState } from "react";

const links = [
  { href: "/", label: "Home" },
  { href: "/dashboard", label: "Dashboard" },
  { href: "/watchlist", label: "Watchlist" },
  { href: "/about", label: "About" },
];

export function Navbar() {
  const pathname = usePathname();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  return (
    <header className="sticky top-0 z-50 backdrop-blur-xl bg-background/60 border-b border-primary/20">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 sm:px-6 py-3">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-3 group">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors duration-300">
            <TrendingUp className="w-5 h-5 text-primary" />
          </div>

          <div className="flex flex-col leading-tight">
            <span className="text-lg font-bold tracking-tight text-primary">
              FinTech Forecaster
            </span>
            <span className="text-[11px] font-medium text-muted-foreground hidden sm:block">
              AI Market Intelligence
            </span>
          </div>
        </Link>

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center gap-6">
          {links.map((link) => {
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  "relative text-sm font-medium transition-colors duration-200",
                  isActive
                    ? "text-primary"
                    : "text-muted-foreground hover:text-primary/80"
                )}
              >
                {link.label}
                {isActive && (
                  <span className="absolute -bottom-1 left-0 w-full h-0.5 bg-gradient-to-r from-primary/70 via-primary to-primary/70 rounded-full" />
                )}
              </Link>
            );
          })}
        </nav>

        {/* Mobile menu button */}
        <div className="md:hidden">
          <button
            onClick={toggleMobileMenu}
            className="inline-flex items-center justify-center p-2 rounded-md text-muted-foreground hover:text-primary hover:bg-primary/10 focus:outline-none focus:ring-2 focus:ring-primary/20 transition-colors duration-200"
            aria-expanded={isMobileMenuOpen}
            aria-label="Toggle mobile menu"
          >
            {isMobileMenuOpen ? (
              <X className="h-6 w-6" />
            ) : (
              <Menu className="h-6 w-6" />
            )}
          </button>
        </div>
      </div>

      {/* Mobile Navigation Menu */}
      {isMobileMenuOpen && (
        <div className="md:hidden border-t border-primary/20 bg-background/95 backdrop-blur-xl">
          <div className="px-4 py-3 space-y-3">
            {links.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setIsMobileMenuOpen(false)}
                  className={cn(
                    "block px-3 py-2 rounded-md text-base font-medium transition-colors duration-200",
                    isActive
                      ? "text-primary bg-primary/10"
                      : "text-muted-foreground hover:text-primary hover:bg-primary/5"
                  )}
                >
                  {link.label}
                </Link>
              );
            })}
          </div>
        </div>
      )}
    </header>
  );
}
