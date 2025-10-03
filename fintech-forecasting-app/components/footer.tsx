import Link from "next/link";

export function Footer() {
  return (
    <footer className="border-t border-border/60 bg-background/75 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl flex-col gap-4 px-6 py-8 text-sm text-muted-foreground md:flex-row md:items-center md:justify-between">
        <div>
          <p className="font-medium text-foreground">
            CS4063 - Natural Language Processing
          </p>
          <p>Assignment: FinTech Forecasting App · Due Oct 7, 10:00 AM</p>
        </div>
        <div className="flex items-center gap-4">
          <Link
            href="https://www.alphavantage.co/documentation/"
            target="_blank"
            rel="noopener noreferrer"
            className="transition-colors hover:text-foreground"
          >
            Alpha Vantage API
          </Link>
          <Link
            href="https://nextjs.org/"
            target="_blank"
            rel="noopener noreferrer"
            className="transition-colors hover:text-foreground"
          >
            Built with Next.js 15
          </Link>
        </div>
      </div>
    </footer>
  );
}
