'use client';'use client';



import { useState, useEffect } from 'react';import { useState, useEffect, useCallback, useMemo } from 'react';

import Link from 'next/link';import Link from 'next/link';

import { Button } from '@/components/ui/button';import { Line } from 'react-chartjs-2';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';import {

import { Badge } from '@/components/ui/badge';  Chart as ChartJS,

import { TrendingUp, RefreshCw, BarChart3 } from 'lucide-react';  CategoryScale,

  LinearScale,

export default function ForecastPage() {  PointElement,

  const [symbol, setSymbol] = useState('AAPL');  LineElement,

  const [isLoading, setIsLoading] = useState(false);  Title,

  const [forecastData, setForecastData] = useState<any>(null);  Tooltip,

  Legend,

  const generateForecast = async () => {  TimeScale,

    setIsLoading(true);  ChartOptions

    try {} from 'chart.js';

      const response = await fetch('/api/forecast', {import 'chartjs-adapter-date-fns';

        method: 'POST',import { Button } from '@/components/ui/button';

        headers: { 'Content-Type': 'application/json' },import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

        body: JSON.stringify({import { Badge } from '@/components/ui/badge';

          symbol,import { 

          model_type: 'lstm',  TrendingUp, 

          horizon: 24,  RefreshCw, 

          retrain: false  Calendar, 

        })  Zap, 

      });  BarChart3 

      const data = await response.json();} from 'lucide-react';

      setForecastData(data);

    } catch (error) {// Register Chart.js components

      console.error('Forecast error:', error);ChartJS.register(

    } finally {  CategoryScale,

      setIsLoading(false);  LinearScale,

    }  PointElement,

  };  LineElement,

  Title,

  return (  Tooltip,

    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">  Legend,

      <div className="container mx-auto px-4 py-8">  TimeScale

        <div className="mb-8">);

          <div className="flex items-center justify-between mb-6">

            <div>// Types

              <h1 className="text-4xl font-bold text-white mb-2">interface HistoricalDataPoint {

                AI Forecasting  timestamp: string;

              </h1>  open: number;

              <p className="text-gray-300">  high: number;

                Advanced machine learning predictions for financial markets  low: number;

              </p>  close: number;

            </div>  volume: number;

            <Link href="/dashboard">}

              <Button variant="outline" className="text-white border-gray-600 hover:bg-gray-800">

                Back to Dashboardinterface ForecastPoint {

              </Button>  timestamp: string;

            </Link>  value: number;

          </div>}



          <div className="flex items-center gap-4 mb-6">interface ForecastData {

            <input  symbol: string;

              type="text"  model: string;

              value={symbol}  forecast: ForecastPoint[];

              onChange={(e) => setSymbol(e.target.value.toUpperCase())}  metrics?: {

              placeholder="Enter symbol (e.g., AAPL)"    mse?: number;

              className="px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:border-blue-500 focus:outline-none"    mae?: number;

            />    rmse?: number;

            <Button   };

              onClick={generateForecast}}

              disabled={isLoading}

              className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3"const models = [

            >  { id: 'moving_average', name: 'Moving Average', color: '#3B82F6', traditional: true },

              {isLoading ? <RefreshCw className="h-4 w-4 animate-spin" /> : "Generate Forecast"}  { id: 'arima', name: 'ARIMA', color: '#10B981', traditional: true },

            </Button>  { id: 'lstm', name: 'LSTM', color: '#F59E0B', traditional: false },

          </div>  { id: 'gru', name: 'GRU', color: '#EF4444', traditional: false },

        </div>  { id: 'transformer', name: 'Transformer', color: '#8B5CF6', traditional: false },

];

        <Card className="bg-gray-800/50 border-gray-700">

          <CardHeader>const timeRanges = [

            <CardTitle className="text-white flex items-center gap-2">  { id: '1h', name: '1 Hour', hours: 1 },

              <BarChart3 className="h-5 w-5" />  { id: '6h', name: '6 Hours', hours: 6 },

              {symbol} Price Forecast  { id: '24h', name: '1 Day', hours: 24 },

            </CardTitle>  { id: '7d', name: '1 Week', hours: 168 },

          </CardHeader>];

          <CardContent>

            {forecastData ? (export default function ForecastPage() {

              <div className="text-white">  const [symbol, setSymbol] = useState('AAPL');

                <h3 className="text-xl mb-4">Forecast Results</h3>  const [selectedModel, setSelectedModel] = useState('lstm');

                <Badge className="mb-4">Model: {forecastData.model || 'LSTM'}</Badge>  const [selectedRange, setSelectedRange] = useState('24h');

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">  const [historicalData, setHistoricalData] = useState<HistoricalDataPoint[]>([]);

                  {forecastData.forecast?.slice(0, 5).map((point: any, idx: number) => (  const [forecastData, setForecastData] = useState<ForecastData | null>(null);

                    <div key={idx} className="bg-gray-700 p-4 rounded-lg">  const [isLoading, setIsLoading] = useState(false);

                      <div className="text-sm text-gray-400">  const [error, setError] = useState<string | null>(null);

                        {new Date(point.timestamp).toLocaleString()}

                      </div>  const fetchHistoricalData = useCallback(async (symbolToFetch: string) => {

                      <div className="text-lg font-bold">    try {

                        ${point.value?.toFixed(2) || 'N/A'}      setIsLoading(true);

                      </div>      setError(null);

                    </div>      

                  ))}      const response = await fetch(`/api/alpha/${symbolToFetch}`);

                </div>      if (!response.ok) throw new Error("Failed to fetch data");

              </div>      

            ) : (      const data = await response.json();

              <div className="text-center text-gray-400 py-12">      if (data.timeseries) {

                <BarChart3 className="h-16 w-16 mx-auto mb-4 opacity-50" />        setHistoricalData(data.timeseries.slice(0, 50));

                <p className="text-lg">Enter a symbol and generate forecast to see predictions</p>      }

              </div>    } catch (err) {

            )}      setError(err instanceof Error ? err.message : "Failed to fetch data");

          </CardContent>    } finally {

        </Card>      setIsLoading(false);

      </div>    }

    </div>  }, []);

  );

}  const generateForecastData = useCallback(async () => {
    if (!symbol) return;
    
    try {
      setIsLoading(true);
      setError(null);
      
      const selectedHours = timeRanges.find(r => r.id === selectedRange)?.hours || 168;
      
      const response = await fetch('/api/forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol,
          model_type: selectedModel,
          horizon: selectedHours,
          retrain: false
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const forecast = await response.json();
      setForecastData(forecast);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate forecast");
    } finally {
      setIsLoading(false);
    }
  }, [symbol, selectedModel, selectedRange]);

  useEffect(() => {
    if (symbol) {
      fetchHistoricalData(symbol);
    }
  }, [symbol, fetchHistoricalData]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">
                AI Forecasting
              </h1>
              <p className="text-gray-300">
                Advanced machine learning predictions for financial markets
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="outline" className="text-white border-gray-600 hover:bg-gray-800">
                Back to Dashboard
              </Button>
            </Link>
          </div>
        </div>
        <Card className="bg-gray-800/50 border-gray-700">
          <CardContent className="p-8">
            <div className="text-center text-white">
              <h2 className="text-2xl font-bold mb-4">Interactive Forecast Charts Coming Soon!</h2>
              <p className="text-gray-300">We are implementing Chart.js integration for beautiful forecast visualization.</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
