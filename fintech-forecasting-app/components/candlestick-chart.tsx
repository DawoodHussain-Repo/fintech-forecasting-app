import React from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  TimeScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import { Chart } from "react-chartjs-2";
import "chartjs-adapter-date-fns";

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  TimeScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface PriceData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface ForecastPoint {
  timestamp: string;
  value: number;
}

interface Props {
  historicalData: PriceData[];
  forecastData: ForecastPoint[];
  symbol: string;
  loading?: boolean;
}

export default function CandlestickChart({
  historicalData,
  forecastData,
  symbol,
  loading = false,
}: Props) {
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: "index" as const,
      intersect: false,
    },
    plugins: {
      legend: {
        position: "top" as const,
        labels: {
          color: "#e5e7eb",
          font: {
            size: 12,
          },
        },
      },
      title: {
        display: true,
        text: `${symbol} - Price & Forecast`,
        color: "#f9fafb",
        font: {
          size: 16,
          weight: "bold" as const,
        },
      },
      tooltip: {
        backgroundColor: "rgba(17, 24, 39, 0.95)",
        titleColor: "#f9fafb",
        bodyColor: "#e5e7eb",
        borderColor: "#374151",
        borderWidth: 1,
        callbacks: {
          label: function (context: {
            dataset: { label?: string };
            parsed: { y: number };
          }) {
            const datasetLabel = context.dataset.label || "";
            const value = context.parsed.y;
            return `${datasetLabel}: $${value.toFixed(2)}`;
          },
        },
      },
    },
    scales: {
      x: {
        type: "time" as const,
        time: {
          unit: "day" as const,
          displayFormats: {
            day: "MMM dd",
          },
        },
        grid: {
          color: "#374151",
        },
        ticks: {
          color: "#e5e7eb",
        },
      },
      y: {
        grid: {
          color: "#374151",
        },
        ticks: {
          color: "#e5e7eb",
          callback: function (value: number | string) {
            return "$" + Number(value).toFixed(2);
          },
        },
      },
    },
  };

  // Prepare data for Chart.js
  const chartData = {
    datasets: [
      // Candlestick data (using line chart as approximation)
      {
        label: "Close Price",
        data: historicalData.map((item) => ({
          x: new Date(item.timestamp),
          y: item.close,
        })),
        borderColor: "#3b82f6",
        backgroundColor: "rgba(59, 130, 246, 0.1)",
        fill: false,
        tension: 0.1,
        pointRadius: 0,
        pointHoverRadius: 4,
        borderWidth: 2,
      },
      // High-Low range
      {
        label: "High",
        data: historicalData.map((item) => ({
          x: new Date(item.timestamp),
          y: item.high,
        })),
        borderColor: "rgba(34, 197, 94, 0.3)",
        backgroundColor: "transparent",
        fill: false,
        pointRadius: 0,
        borderWidth: 1,
        borderDash: [2, 2],
      },
      {
        label: "Low",
        data: historicalData.map((item) => ({
          x: new Date(item.timestamp),
          y: item.low,
        })),
        borderColor: "rgba(239, 68, 68, 0.3)",
        backgroundColor: "transparent",
        fill: false,
        pointRadius: 0,
        borderWidth: 1,
        borderDash: [2, 2],
      },
      // Forecast data
      ...(forecastData.length > 0
        ? [
            {
              label: "Forecast",
              data: forecastData.map((item) => ({
                x: new Date(item.timestamp),
                y: item.value,
              })),
              borderColor: "#f59e0b",
              backgroundColor: "rgba(245, 158, 11, 0.1)",
              fill: false,
              tension: 0.1,
              pointRadius: 3,
              pointHoverRadius: 6,
              borderWidth: 3,
              borderDash: [5, 5],
            },
          ]
        : []),
    ],
  };

  if (loading) {
    return (
      <div className="h-96 bg-gray-800 rounded-lg flex items-center justify-center">
        <div className="flex items-center space-x-2">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="text-gray-300">Loading chart...</span>
        </div>
      </div>
    );
  }

  if (!historicalData || historicalData.length === 0) {
    return (
      <div className="h-96 bg-gray-800 rounded-lg flex items-center justify-center">
        <div className="text-center">
          <div className="text-6xl mb-4">📈</div>
          <p className="text-gray-400">No data available</p>
          <p className="text-sm text-gray-500">
            Try selecting a different symbol
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="h-96">
        <Chart type="line" data={chartData} options={options} />
      </div>

      {/* Chart Info */}
      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div className="bg-gray-700 rounded p-3">
          <div className="text-gray-400">Data Points</div>
          <div className="text-white font-semibold">
            {historicalData.length}
          </div>
        </div>
        <div className="bg-gray-700 rounded p-3">
          <div className="text-gray-400">Forecast Points</div>
          <div className="text-orange-400 font-semibold">
            {forecastData.length}
          </div>
        </div>
        <div className="bg-gray-700 rounded p-3">
          <div className="text-gray-400">Latest Price</div>
          <div className="text-blue-400 font-semibold">
            ${historicalData[historicalData.length - 1]?.close.toFixed(2)}
          </div>
        </div>
        <div className="bg-gray-700 rounded p-3">
          <div className="text-gray-400">Next Forecast</div>
          <div className="text-orange-400 font-semibold">
            ${forecastData[0]?.value.toFixed(2) || "N/A"}
          </div>
        </div>
      </div>
    </div>
  );
}
