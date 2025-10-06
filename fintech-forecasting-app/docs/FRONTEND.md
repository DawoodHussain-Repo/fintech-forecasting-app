# Frontend Documentation

## Overview

Next.js 15 + React 19 application with TypeScript, providing a modern, responsive UI for financial forecasting with real-time stock data visualization.

## Tech Stack

### Core Framework

- **Next.js 15.0.3** - React framework with App Router
- **React 19** - UI library
- **TypeScript** - Type safety

### UI Components

- **Tailwind CSS** - Utility-first styling
- **Chart.js** - Data visualization
- **Lucide React** - Icon library
- **GSAP** - Animations

### Data Fetching

- **yfinance API** - Stock market data via backend
- **Finnhub API** - Real-time quotes

## Project Structure

```
app/
├── layout.tsx              # Root layout with theme provider
├── page.tsx                # Landing page
├── globals.css             # Global styles
├── about/
│   └── page.tsx           # About page
├── dashboard/
│   └── page.tsx           # Main dashboard with stock cards
├── stock/
│   └── [symbol]/
│       ├── page.tsx       # Stock detail page
│       └── loading.tsx    # Loading state
├── watchlist/
│   └── page.tsx          # Watchlist management
└── api/
    ├── forecast/
    │   └── route.ts      # Forecast API proxy
    ├── finnhub/
    │   └── [symbol]/
    │       └── route.ts  # Finnhub data proxy
    └── models/
        └── route.ts      # Available models list

components/
├── navbar.tsx                    # Navigation header
├── footer.tsx                    # Footer component
├── theme-toggle.tsx             # Dark/light mode toggle
├── search-bar.tsx               # Stock search
├── candlestick-chart.tsx        # Price chart visualization
├── forecast-card.tsx            # Forecast display
├── metric-card.tsx              # Metric display
├── model-comparison.tsx         # Model comparison table
├── liquid-glass-stock-card.tsx  # Stock card with glass effect
└── ui/
    ├── button.tsx              # Button component
    ├── card.tsx                # Card component
    └── input.tsx               # Input component

lib/
├── types.ts              # TypeScript interfaces
├── utils.ts              # Utility functions
├── alpha-vantage.ts      # Market data fetching
└── forecast.ts           # Forecast utilities

hooks/
├── use-watchlist.ts      # Watchlist management
└── use-local-storage.ts  # Local storage hook
```

## Key Features

### 1. Stock Search & Dashboard

- Real-time stock search
- Popular stocks display
- Watchlist management
- Quick access to forecasts

### 2. Stock Detail Page

**Route**: `/stock/[symbol]`

Features:

- Real-time price data
- OHLCV statistics
- Interactive candlestick chart
- Multiple timeframe selection (1D, 1W, 1M, 6M, 1Y)
- AI forecasting with multiple models

### 3. Forecasting Interface

Forecast horizons: **6h, 12h, 24h, 48h, 72h** (1-72 hours only)

Available models:

- Moving Average
- ARIMA
- LSTM
- GRU
- Transformer

**Forecast Display:**

```typescript
interface ForecastData {
  symbol: string;
  model_type: string;
  current_price: number;
  forecast_1h: ShortTermForecast;
  forecast_4h: ShortTermForecast;
  forecast_24h: ShortTermForecast;
  technical_indicators: TechnicalIndicators;
  forecast: BackendForecastPoint[];
  data_points_used: number;
  data_period: string;
  created_at: string;
  cached: boolean;
}
```

### 4. Chart Visualization

**Component**: `candlestick-chart.tsx`

Features:

- Candlestick chart for historical data
- Line overlay for forecast predictions
- Responsive design
- Dynamic scaling
- Hover tooltips

### 5. Dark Mode

- System preference detection
- Manual toggle
- Persistent across sessions
- Smooth transitions

## API Integration

### Backend Endpoints Used

#### 1. Forecast Generation

```typescript
POST / api / forecast;
Body: {
  symbol: string;
  model_type: "moving_average" | "arima" | "lstm" | "gru" | "transformer";
  horizon: number; // 1-72 hours
}
Response: ForecastData;
```

#### 2. Stock Data

```typescript
GET /api/finnhub/[symbol]
Response: {
  quote: {
    price: number;
    changePercent: number;
    high: number;
    low: number;
    open: number;
    previousClose: number;
    latestTradingDay: string;
  };
  candles?: PriceData[];
}
```

#### 3. Available Models

```typescript
GET / api / models;
Response: {
  models: Array<{
    type: string;
    name: string;
    description: string;
    traditional: boolean;
  }>;
}
```

## Type Definitions

### Core Types

```typescript
interface BackendForecastPoint {
  timestamp: string;
  predicted_price: number;
  price_range_low: number;
  price_range_high: number;
  confidence: string;
}

interface ShortTermForecast {
  direction: string; // 'up' | 'down' | 'sideways'
  confidence: string; // 'high' | 'medium' | 'low'
  predicted_price: number;
  price_range: [number, number];
}

interface TechnicalIndicators {
  sma_7: number;
  rsi: number;
  bb_upper: number;
  bb_lower: number;
  volatility_pct: number;
  momentum_5d_pct: number;
  support_level: number;
  resistance_level: number;
}

interface PriceData {
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}
```

## Styling System

### Tailwind Configuration

```typescript
// tailwind.config.ts
- Custom color scheme
- Dark mode support
- Responsive breakpoints
- Custom animations
```

### Glass Effect

Used throughout UI for modern aesthetic:

```css
.glass {
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
}
```

## Environment Variables

```bash
# Frontend (.env.local)
BACKEND_URL=http://localhost:5000
NEXT_PUBLIC_FINNHUB_API_KEY=your_key_here
```

## Development

### Installation

```bash
npm install
```

### Run Development Server

```bash
npm run dev
```

Visit: http://localhost:3000

### Build for Production

```bash
npm run build
npm start
```

### Linting

```bash
npm run lint
```

## Component Usage Examples

### Using Forecast Component

```tsx
import { useState } from "react";

const [forecastData, setForecastData] = useState<ForecastData | null>(null);

const handleGenerateForecast = async (modelType: string) => {
  const response = await fetch("/api/forecast", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      symbol: "AAPL",
      model_type: modelType,
      horizon: 24,
    }),
  });
  const result = await response.json();
  setForecastData(result);
};
```

### Using Chart Component

```tsx
<CandlestickChart
  historicalData={priceData}
  forecastData={
    forecastData?.forecast.map((p) => ({
      timestamp: p.timestamp,
      value: p.predicted_price,
    })) || []
  }
  symbol="AAPL"
  loading={false}
/>
```

## Performance Optimizations

1. **Code Splitting**: Automatic with Next.js App Router
2. **Image Optimization**: Next.js Image component
3. **Lazy Loading**: React.lazy for heavy components
4. **Memoization**: useMemo and useCallback for expensive computations
5. **Server Components**: Default in App Router for better performance

## Best Practices

1. **Type Safety**: Use TypeScript interfaces for all data structures
2. **Error Handling**: Always wrap API calls in try-catch blocks
3. **Loading States**: Show loading indicators during data fetching
4. **Responsive Design**: Mobile-first approach with Tailwind
5. **Accessibility**: Semantic HTML and ARIA labels

## Troubleshooting

### Chart Not Displaying

- Ensure data is in correct format
- Check if historical data exists
- Verify Chart.js is loaded

### API Errors

- Check backend is running on correct port
- Verify BACKEND_URL in environment variables
- Check CORS settings

### TypeScript Errors

- Run `npm run type-check`
- Ensure all types are properly imported
- Check tsconfig.json settings

## Future Enhancements

- [ ] WebSocket for real-time price updates
- [ ] Portfolio management
- [ ] Advanced charting (indicators, drawings)
- [ ] Price alerts
- [ ] Mobile app (React Native)
- [ ] Multi-language support
