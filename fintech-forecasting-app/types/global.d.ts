declare module "*.css";

// Additional module declarations
declare module "chartjs-chart-financial" {
  export * from "chart.js";
}

declare module "chartjs-adapter-date-fns" {
  export * from "chart.js";
}

// Global types for the application
interface Window {
  // Add any global window properties here
}

// Financial data types
interface PriceData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface ForecastData {
  timestamp: string;
  value: number;
  confidence?: {
    lower: number;
    upper: number;
  };
}

interface ModelMetrics {
  mse: number;
  mae: number;
  rmse: number;
  mape?: number;
  accuracy?: number;
}
