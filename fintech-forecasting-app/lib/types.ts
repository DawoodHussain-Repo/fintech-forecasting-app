export interface Candle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Quote {
  symbol: string;
  price: number;
  changePercent: number;
  open: number;
  high: number;
  low: number;
  previousClose: number;
  volume: number;
  latestTradingDay: string;
  currency?: string;
}

export interface MarketSnapshot {
  quote: Quote | null;
  candles: Candle[];
}
