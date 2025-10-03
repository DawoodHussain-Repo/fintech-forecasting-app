import Link from "next/link";

const highlights = [
  {
    title: "AI-ready architecture",
    description:
      "Plug in forecasting models like ARIMA, LSTM, or Transformers when your ML pipeline is ready.",
  },
  {
    title: "Multi-asset coverage",
    description:
      "Track equities, crypto tokens, and forex pairs in one unified workspace.",
  },
  {
    title: "Alpha Vantage integration",
    description:
      "Instant data sync with interactive candlestick charts and watchlists.",
  },
];

export default function LandingPage() {
  return (
    <div className="relative isolate overflow-hidden rounded-3xl border border-border/60 bg-background/60 p-10 shadow-xl shadow-black/5">
      <div className="flex flex-col gap-12 lg:grid lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:items-center">
        <div className="space-y-8">
          <div className="inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-4 py-1 text-sm font-medium text-primary">
            FinTech Forecaster
            <span className="text-muted-foreground">v0.1</span>
          </div>
          <h1 className="text-4xl font-semibold tracking-tight text-foreground sm:text-5xl">
            Forecast stocks, crypto, and forex with AI-powered insights (coming
            soon)
          </h1>
          <p className="max-w-xl text-lg text-muted-foreground">
            Explore market momentum with a minimal, Apple-inspired dashboard.
            Fetch live data from Alpha Vantage and be ready to plug in your
            machine learning forecasts as the assignment evolves.
          </p>
          <div className="flex flex-col gap-3 sm:flex-row">
            <Link
              href="/dashboard"
              className="inline-flex items-center justify-center rounded-full bg-foreground px-8 py-3 text-sm font-medium text-background shadow hover:bg-foreground/90"
            >
              Go to Dashboard
            </Link>
            <Link
              href="/dashboard"
              className="inline-flex items-center justify-center rounded-full border border-border/60 px-8 py-3 text-sm font-medium text-foreground hover:bg-muted/40"
            >
              Login (placeholder)
            </Link>
          </div>
        </div>
        <div className="grid gap-4">
          {highlights.map((item) => (
            <div
              key={item.title}
              className="rounded-2xl border border-border/50 bg-card/70 p-6 shadow-sm backdrop-blur"
            >
              <h2 className="text-lg font-semibold text-foreground">
                {item.title}
              </h2>
              <p className="mt-2 text-sm text-muted-foreground">
                {item.description}
              </p>
            </div>
          ))}
        </div>
      </div>

      <div className="pointer-events-none absolute -top-24 right-0 h-64 w-64 rounded-full bg-primary/30 blur-[120px]" />
      <div className="pointer-events-none absolute bottom-0 left-10 h-56 w-56 rounded-full bg-secondary/20 blur-[110px]" />
    </div>
  );
}
