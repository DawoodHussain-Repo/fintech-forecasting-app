import Link from "next/link";
import { TrendingUp, Globe } from "lucide-react";

export function Footer() {
  return (
    <footer className="glass mt-20 border-t border-primary/20">
      <div className="mx-auto max-w-7xl px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
          {/* Brand */}
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-primary/20 to-primary/5 border border-primary/30 flex items-center justify-center">
                <TrendingUp className="w-5 h-5 text-primary" />
              </div>
              <span className="text-xl font-bold text-primary text-glow">
                FinTech Forecaster
              </span>
            </div>
            <p className="text-sm text-muted-foreground">
              AI-powered market intelligence and forecasting platform for modern
              traders.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-sm font-semibold text-primary mb-4">
              Quick Links
            </h3>
            <div className="flex flex-col gap-2 text-sm">
              <Link
                href="/dashboard"
                className="text-muted-foreground hover:text-primary transition-colors"
              >
                Dashboard
              </Link>
              <Link
                href="/forecast"
                className="text-muted-foreground hover:text-primary transition-colors"
              >
                Forecast
              </Link>
              <Link
                href="/watchlist"
                className="text-muted-foreground hover:text-primary transition-colors"
              >
                Watchlist
              </Link>
              <Link
                href="/about"
                className="text-muted-foreground hover:text-primary transition-colors"
              >
                About
              </Link>
            </div>
          </div>

          {/* Social */}
          <div>
            <h3 className="text-sm font-semibold text-primary mb-4">Connect</h3>
            <div className="flex gap-3">
              <a
                href="https://github.com/DawoodHussain-Repo/fintech-forecasting-app"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 glass hover:bg-primary/10 rounded-lg transition-all hover:glow-sm"
              >
                <Globe className="w-5 h-5 text-muted-foreground hover:text-primary transition-colors" />
              </a>
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="pt-8 border-t border-primary/10 flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-muted-foreground">
          <p>
            &copy; {new Date().getFullYear()} FinTech Forecaster. All rights
            reserved.
          </p>
          <p className="text-xs">Powered by Finnhub API & Advanced ML Models</p>
        </div>
      </div>

      {/* Decorative bottom line */}
      <div className="h-0.5 bg-gradient-to-r from-transparent via-primary to-transparent" />
    </footer>
  );
}
