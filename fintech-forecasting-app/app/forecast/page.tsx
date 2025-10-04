"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { TrendingUp, RefreshCw } from "lucide-react";

export default function ForecastPage() {
  const [symbol, setSymbol] = useState("AAPL");
  const [isLoading, setIsLoading] = useState(false);
  const [forecastData, setForecastData] = useState<any>(null);

  const generateForecast = async () => {
    setIsLoading(true);
    try {
      const response = await fetch("/api/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol,
          model_type: "lstm",
          horizon: 7,
          retrain: false,
        }),
      });
      const result = await response.json();
      setForecastData(result);
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Stock Forecast</h1>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Generate Forecast</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-4">
            <Input
              placeholder="Enter stock symbol (e.g., AAPL)"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              className="flex-1"
            />
            <Button
              onClick={generateForecast}
              disabled={isLoading}
              className="flex items-center gap-2"
            >
              {isLoading ? (
                <RefreshCw className="h-4 w-4 animate-spin" />
              ) : (
                <TrendingUp className="h-4 w-4" />
              )}
              {isLoading ? "Generating..." : "Generate Forecast"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {forecastData && (
        <Card>
          <CardHeader>
            <CardTitle>Forecast Results</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="text-sm bg-gray-100 p-4 rounded">
              {JSON.stringify(forecastData, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
