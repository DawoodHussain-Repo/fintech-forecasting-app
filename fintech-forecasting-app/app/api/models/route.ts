import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:5000";

export async function GET() {
  try {
    // Forward request to Python backend to get available models
    const response = await fetch(`${BACKEND_URL}/api/models`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      // Fallback to local model list if backend is unavailable
      return NextResponse.json({
        models: [
          {
            type: "moving_average",
            name: "Moving Average",
            description: "Simple moving average forecast",
            traditional: true,
          },
          {
            type: "arima",
            name: "ARIMA",
            description: "AutoRegressive Integrated Moving Average",
            traditional: true,
          },
          {
            type: "lstm",
            name: "LSTM",
            description: "Long Short-Term Memory neural network",
            traditional: false,
          },
          {
            type: "gru",
            name: "GRU",
            description: "Gated Recurrent Unit neural network",
            traditional: false,
          },
          {
            type: "transformer",
            name: "Transformer",
            description: "Transformer-based neural network",
            traditional: false,
          },
        ],
      });
    }

    const modelsData = await response.json();
    return NextResponse.json(modelsData);
  } catch (error) {
    console.error("Models API error:", error);

    // Return fallback models list
    return NextResponse.json({
      models: [
        {
          type: "lstm",
          name: "LSTM",
          description: "Long Short-Term Memory neural network (fallback)",
          traditional: false,
        },
        {
          type: "arima",
          name: "ARIMA",
          description: "AutoRegressive Integrated Moving Average (fallback)",
          traditional: true,
        },
      ],
    });
  }
}
