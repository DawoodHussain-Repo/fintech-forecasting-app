import Link from "next/link";

const highlights = [
  {
    title: "🤖 AI-Powered Forecasting",
    description:
      "Advanced ML models including ARIMA, LSTM, and Transformer networks for accurate market predictions.",
  },
  {
    title: "🌍 Multi-Asset Coverage",
    description:
      "Track stocks, crypto, and forex with real-time data and comprehensive analytics.",
  },
  {
    title: "📊 Interactive Analytics",
    description:
      "Professional candlestick charts with technical indicators and portfolio management tools.",
  },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 to-purple-600/20" />
        <div className="relative mx-auto max-w-7xl px-6 py-24 sm:py-32 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <div className="mb-8">
              <span className="inline-flex items-center gap-2 rounded-full bg-blue-500/10 px-4 py-2 text-sm font-medium text-blue-400 ring-1 ring-blue-500/20">
                FinTech Forecaster
                <span className="text-blue-300">v1.0</span>
              </span>
            </div>
            <h1 className="text-4xl font-bold tracking-tight text-white sm:text-6xl">
              AI-Powered
              <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                {" "}
                Financial{" "}
              </span>
              Forecasting
            </h1>
            <p className="mt-6 text-lg leading-8 text-gray-300">
              Advanced machine learning models for stock, crypto, and forex
              prediction. Real-time market data with professional-grade
              analytics and forecasting tools.
            </p>
            <div className="mt-10 flex items-center justify-center gap-x-6">
              <Link
                href="/dashboard"
                className="group inline-flex items-center gap-2 rounded-lg bg-blue-600 px-6 py-3 text-sm font-semibold text-white shadow-lg hover:bg-blue-500 transition-colors"
              >
                Launch Dashboard
                <span className="transition-transform group-hover:translate-x-1">
                  →
                </span>
              </Link>
              <Link
                href="/forecast"
                className="inline-flex items-center gap-2 rounded-lg border border-gray-600 px-6 py-3 text-sm font-semibold text-gray-300 hover:bg-gray-800 hover:text-white transition-colors"
              >
                Try Forecasting
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-24 sm:py-32">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
              Professional Trading Tools
            </h2>
            <p className="mt-6 text-lg leading-8 text-gray-300">
              Everything you need for advanced financial market analysis and
              prediction.
            </p>
          </div>
          <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none">
            <dl className="grid max-w-xl grid-cols-1 gap-x-8 gap-y-16 lg:max-w-none lg:grid-cols-3">
              {highlights.map((feature) => (
                <div key={feature.title} className="flex flex-col">
                  <dt className="text-base font-semibold leading-7 text-white mb-2">
                    {feature.title}
                  </dt>
                  <dd className="mt-1 flex flex-auto flex-col text-base leading-7 text-gray-300">
                    <p className="flex-auto">{feature.description}</p>
                  </dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="relative isolate mt-32 px-6 py-32 sm:mt-56 sm:py-40 lg:px-8">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/10 to-purple-600/10" />
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Ready to start forecasting?
          </h2>
          <p className="mx-auto mt-6 max-w-xl text-lg leading-8 text-gray-300">
            Access real-time market data and AI-powered predictions for informed
            trading decisions.
          </p>
          <div className="mt-10 flex items-center justify-center gap-x-6">
            <Link
              href="/dashboard"
              className="rounded-lg bg-blue-600 px-6 py-3 text-sm font-semibold text-white shadow-lg hover:bg-blue-500 transition-colors"
            >
              Get started
            </Link>
            <Link
              href="/about"
              className="text-sm font-semibold leading-6 text-gray-300 hover:text-white transition-colors"
            >
              Learn more <span aria-hidden={true}>→</span>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
