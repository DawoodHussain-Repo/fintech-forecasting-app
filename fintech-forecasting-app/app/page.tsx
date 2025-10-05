"use client";

import Link from "next/link";
import { useEffect, useRef } from "react";
import { gsap } from "gsap";
import { TrendingUp, Activity, BarChart3, ChevronRight } from "lucide-react";

const features = [
  {
    icon: Activity,
    title: "AI-Powered Forecasting",
    description:
      "Advanced ML models including ARIMA, LSTM, and Transformer networks for accurate predictions.",
  },
  {
    icon: TrendingUp,
    title: "Multi-Asset Coverage",
    description:
      "Track stocks, crypto, and forex with real-time data and comprehensive analytics.",
  },
  {
    icon: BarChart3,
    title: "Interactive Analytics",
    description:
      "Professional charts with technical indicators and portfolio management tools.",
  },
];

export default function HomePage() {
  const heroRef = useRef<HTMLDivElement>(null);
  const featuresRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (heroRef.current) {
      gsap.fromTo(
        heroRef.current.children,
        { opacity: 0, y: 30 },
        {
          opacity: 1,
          y: 0,
          duration: 0.8,
          stagger: 0.2,
          ease: "power3.out",
        }
      );
    }
  }, []);

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-[120px] animate-pulse" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-primary/5 rounded-full blur-[120px] animate-pulse" style={{ animationDelay: "1s" }} />
      </div>

      <div className="relative z-10">
        {/* Hero Section */}
        <div className="container mx-auto px-4 pt-20 pb-32">
          <div ref={heroRef} className="max-w-4xl mx-auto text-center space-y-8">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-4 py-2 glass border border-primary/20 rounded-full">
              <div className="w-2 h-2 bg-primary rounded-full animate-pulse" />
              <span className="text-sm text-primary font-medium">
                AI-Powered Trading Platform
              </span>
            </div>

            {/* Main Title */}
            <h1 className="text-5xl md:text-7xl font-bold text-primary leading-tight">
              Financial Forecasting
              <br />
              <span className="text-foreground">Redefined</span>
            </h1>

            {/* Subtitle */}
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Advanced machine learning models for stock, crypto, and forex prediction.
              Real-time market data with professional-grade analytics.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-8">
              <Link
                href="/dashboard"
                className="group px-8 py-3  bg-primary text-green-500 font-semibold rounded-lg transition-all duration-300  inline-flex items-center gap-2"
              >
                Launch Dashboard
                <ChevronRight className="w-4 h-4  transition-transform group-hover:translate-x-1" />
              </Link>
              
              <Link
                href="/forecast"
                className="px-8 py-3 glass border border-primary/20 text-primary font-semibold rounded-lg transition-all duration-300 hover:border-primary/40"
              >
                Try Forecasting
              </Link>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-8 pt-16 max-w-2xl mx-auto">
              <div className="space-y-1">
                <div className="text-3xl font-bold text-primary">5+</div>
                <div className="text-sm text-muted-foreground">ML Models</div>
              </div>
              <div className="space-y-1">
                <div className="text-3xl font-bold text-primary">24/7</div>
                <div className="text-sm text-muted-foreground">Live Data</div>
              </div>
              <div className="space-y-1">
                <div className="text-3xl font-bold text-primary">1000+</div>
                <div className="text-sm text-muted-foreground">Assets</div>
              </div>
            </div>
          </div>
        </div>

        {/* Features Section */}
        <div className="container mx-auto px-4 py-20">
          <div className="max-w-6xl mx-auto">
            <div className="text-center mb-16">
              <h2 className="text-3xl md:text-4xl font-bold text-primary mb-4">
                Professional Trading Tools
              </h2>
              <p className="text-lg text-muted-foreground">
                Everything you need for advanced financial market analysis
              </p>
            </div>

            <div
              ref={featuresRef}
              className="grid md:grid-cols-3 gap-6"
            >
              {features.map((feature, index) => {
                const Icon = feature.icon;
                return (
                  <div
                    key={feature.title}
                    className="glass p-6 space-y-4 hover:border-primary/30 transition-all"
                  >
                    <div className="w-12 h-12 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center">
                      <Icon className="w-6 h-6 text-primary" />
                    </div>
                    <h3 className="text-xl font-bold text-foreground">
                      {feature.title}
                    </h3>
                    <p className="text-muted-foreground">
                      {feature.description}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* CTA Section */}
        <div className="container mx-auto px-4 py-20">
          <div className="max-w-4xl mx-auto glass p-12 text-center space-y-6">
            <h2 className="text-3xl md:text-4xl font-bold text-primary">
              Ready to Start Trading?
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Join thousands of traders using AI-powered insights to make smarter investment decisions.
            </p>
            <div className="pt-6">
              <Link
                href="/dashboard"
                className="inline-flex items-center gap-2 px-8 py-4 bg-primary text-green-500 border-1 !rounded-full border-green-500 font-semibold rounded-lg hover:text-black hover:bg-green-500 transition-all duration-300 hover:shadow-lg hover:shadow-primary/50"
              >
                Get Started Now
                <ChevronRight className="w-5 h-5" />
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
