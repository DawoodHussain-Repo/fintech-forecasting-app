import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:5000";

export async function POST(request: Request) {
  try {
    const {
      symbol,
      model_type = "lstm",
      horizon = 24,
      retrain = false,
    } = await request.json();

    if (!symbol) {
      return NextResponse.json(
        { error: "Symbol is required" },
        { status: 400 }
      );
    }

    // Forward request to Python backend
    const response = await fetch(`${BACKEND_URL}/api/forecast/${symbol}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model_type,
        horizon,
        retrain,
      }),
    });

    if (!response.ok) {
      const errorData = await response
        .json()
        .catch(() => ({ error: "Backend service unavailable" }));
      return NextResponse.json(errorData, { status: response.status });
    }

    const forecastData = await response.json();
    return NextResponse.json(forecastData);
  } catch (error) {
    console.error("Forecast API error:", error);
    return NextResponse.json(
      {
        error: "Failed to generate forecast",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get("symbol");

    if (!symbol) {
      return NextResponse.json(
        { error: "Symbol parameter is required" },
        { status: 400 }
      );
    }

    // Get latest forecast from backend
    const response = await fetch(`${BACKEND_URL}/api/performance/${symbol}`);

    if (!response.ok) {
      const errorData = await response
        .json()
        .catch(() => ({ error: "Backend service unavailable" }));
      return NextResponse.json(errorData, { status: response.status });
    }

    const performanceData = await response.json();
    return NextResponse.json(performanceData);
  } catch (error) {
    console.error("Forecast GET API error:", error);
    return NextResponse.json(
      {
        error: "Failed to get forecast data",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
