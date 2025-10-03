<h1 align="center">FinTech Forecaster</h1>

Minimal, Apple-inspired market analysis dashboard built with **Next.js 15**, **TypeScript**, **Tailwind CSS v4**, and **shadcn/ui**. The app consumes the **Alpha Vantage** API for live stock, crypto, and forex pricing and prepares the UI surface where upcoming ML models (ARIMA, LSTM, Transformers) will plug in.

> 📌 Course: **CS4063 – Natural Language Processing** · Due: **Tuesday, October 7th, 10:00am**

## ✨ Features

- App Router architecture with shared layout, theming, and responsive design.
- Dark/light mode via `next-themes` with a custom Tailwind theme.
- Dashboard with Alpha Vantage integration, candlestick charting (`react-chartjs-2`, `chartjs-chart-financial`), range selector, and fundamentals cards.
- Forecast workspace presenting historical candles plus a dummy forecast overlay—ready for ML model integration.
- Persistent watchlist stored in `localStorage` with sparklines.
- About page summarizing assignment context and motivation.
- API route (`/api/alpha/[symbol]`) that normalizes stock/crypto/forex responses into a unified shape.

## 🚀 Getting Started

### 1. Install dependencies

```bash
npm install
```

### 2. Configure environment

Create a `.env.local` file (or configure environment variables in your deployment platform) with your Alpha Vantage API key. A demo key is used as a fallback but is heavily rate limited.

```bash
ALPHA_VANTAGE_API_KEY=your_key_here
```

### 3. Run locally

```bash
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000) to explore the app. The dashboard defaults to the `AAPL` symbol; try others like `BTC` or `EUR/USD`.

### 4. Lint & build

```bash
npm run lint
npm run build
```

Both commands are CI-friendly and required before deploying to Vercel.

## 🗂️ Project Structure

```
app/
	layout.tsx          # Shared layout, theme provider, navigation, footer
	page.tsx            # Landing page
	dashboard/          # Dashboard UI and data fetching
	forecast/           # Forecast workspace with dummy projections
	watchlist/          # Persisted watchlist table
	about/              # Assignment description & motivation
	api/alpha/[symbol]/ # Serverless route querying Alpha Vantage
components/           # Reusable UI primitives & feature components
hooks/                # Client hooks (local storage + watchlist)
lib/                  # Utilities, formatting helpers, forecasting stubs
styles/               # Design tokens layered into Tailwind
```

## 🔌 Alpha Vantage notes

- API limits (5 requests/min on free tier) are enforced; the route returns a 429 with a helpful message when exceeded.
- Symbol parsing rules:
  - `AAPL`, `MSFT` → stocks (`TIME_SERIES_DAILY_ADJUSTED`)
  - `BTC`, `ETH` → crypto (`DIGITAL_CURRENCY_DAILY`)
  - `EUR/USD`, `GBP/JPY` → forex (`FX_DAILY`)
- The response is normalized server-side so the frontend can consume a consistent `quote` + `candles` payload.

## 🧭 Next Steps (suggested)

- Wire real ML forecasts using your chosen models and display confidence intervals.
- Enrich the dashboard with NLP-derived sentiment indicators.
- Add authentication before exposing portfolio features in production.

Enjoy exploring and extending **FinTech Forecaster**! If you deploy to Vercel, set the `ALPHA_VANTAGE_API_KEY` environment variable in the dashboard for production access.
