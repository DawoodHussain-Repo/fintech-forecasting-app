import React from "react";
import { Badge } from "./ui/badge";

interface ModelMetrics {
  name: string;
  type: "traditional" | "neural";
  mse: number;
  mae: number;
  rmse: number;
  mape: number;
  accuracy?: number;
  trainingTime: number;
  status: "success" | "error" | "loading";
  description: string;
}

interface Props {
  models: ModelMetrics[];
  loading?: boolean;
}

export default function ModelComparison({ models, loading = false }: Props) {
  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-xl font-semibold text-white mb-4">
          Model Performance
        </h3>
        <div className="animate-pulse">
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-16 bg-gray-700 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const getBestModel = (
    metric: keyof Pick<ModelMetrics, "mse" | "mae" | "rmse" | "mape">
  ) => {
    return models.reduce((best, current) =>
      current[metric] < best[metric] ? current : best
    );
  };

  const formatMetric = (
    value: number,
    type: "error" | "time" | "percentage"
  ) => {
    switch (type) {
      case "error":
        return value.toFixed(4);
      case "time":
        return `${value.toFixed(2)}s`;
      case "percentage":
        return `${value.toFixed(2)}%`;
      default:
        return value.toFixed(4);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "success":
        return "bg-green-500/20 text-green-400 border-green-500/30";
      case "error":
        return "bg-red-500/20 text-red-400 border-red-500/30";
      case "loading":
        return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
      default:
        return "bg-gray-500/20 text-gray-400 border-gray-500/30";
    }
  };

  const getTypeColor = (type: string) => {
    return type === "neural"
      ? "bg-purple-500/20 text-purple-400 border-purple-500/30"
      : "bg-blue-500/20 text-blue-400 border-blue-500/30";
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-white">
          Model Performance Comparison
        </h3>
        <div className="flex gap-2">
          <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/30">
            Traditional
          </Badge>
          <Badge className="bg-purple-500/20 text-purple-400 border-purple-500/30">
            Neural Networks
          </Badge>
        </div>
      </div>

      {models.length === 0 ? (
        <div className="text-center py-8">
          <div className="text-6xl mb-4">🤖</div>
          <p className="text-gray-400">No model metrics available</p>
          <p className="text-sm text-gray-500">
            Train some models to see performance data
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Best Performers Summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="text-sm text-gray-400">Lowest MSE</div>
              <div className="text-lg font-semibold text-green-400">
                {getBestModel("mse").name}
              </div>
              <div className="text-sm text-gray-300">
                {formatMetric(getBestModel("mse").mse, "error")}
              </div>
            </div>
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="text-sm text-gray-400">Lowest MAE</div>
              <div className="text-lg font-semibold text-blue-400">
                {getBestModel("mae").name}
              </div>
              <div className="text-sm text-gray-300">
                {formatMetric(getBestModel("mae").mae, "error")}
              </div>
            </div>
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="text-sm text-gray-400">Lowest RMSE</div>
              <div className="text-lg font-semibold text-purple-400">
                {getBestModel("rmse").name}
              </div>
              <div className="text-sm text-gray-300">
                {formatMetric(getBestModel("rmse").rmse, "error")}
              </div>
            </div>
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="text-sm text-gray-400">Lowest MAPE</div>
              <div className="text-lg font-semibold text-orange-400">
                {getBestModel("mape").name}
              </div>
              <div className="text-sm text-gray-300">
                {formatMetric(getBestModel("mape").mape, "percentage")}
              </div>
            </div>
          </div>

          {/* Detailed Model Cards */}
          <div className="grid gap-4">
            {models.map((model, index) => (
              <div key={index} className="bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <h4 className="text-lg font-semibold text-white">
                      {model.name}
                    </h4>
                    <Badge className={getTypeColor(model.type)}>
                      {model.type}
                    </Badge>
                    <Badge className={getStatusColor(model.status)}>
                      {model.status}
                    </Badge>
                  </div>
                  <div className="text-sm text-gray-400">
                    Training: {formatMetric(model.trainingTime, "time")}
                  </div>
                </div>

                <p className="text-sm text-gray-300 mb-4">
                  {model.description}
                </p>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-sm text-gray-400">MSE</div>
                    <div className="text-lg font-semibold text-white">
                      {formatMetric(model.mse, "error")}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-gray-400">MAE</div>
                    <div className="text-lg font-semibold text-white">
                      {formatMetric(model.mae, "error")}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-gray-400">RMSE</div>
                    <div className="text-lg font-semibold text-white">
                      {formatMetric(model.rmse, "error")}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-gray-400">MAPE</div>
                    <div className="text-lg font-semibold text-white">
                      {formatMetric(model.mape, "percentage")}
                    </div>
                  </div>
                </div>

                {model.accuracy && (
                  <div className="mt-3 pt-3 border-t border-gray-600">
                    <div className="text-sm text-gray-400">
                      Accuracy:{" "}
                      <span className="text-green-400 font-semibold">
                        {formatMetric(model.accuracy, "percentage")}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
